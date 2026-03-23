"""
predict_future.py
-----------------
Run the trained DFGCN model on the most-recent `seq_len` rows of your CSV to
produce truly future predictions (beyond the last known data point).

Usage example (one-liner, no backslash continuation needed for Windows):
    python predict_future.py --model_id COP_USD_experimento1 --data_path datos/tasa_cop_usd.csv --seq_len 96 --pred_len 30 --enc_in 1 --d_model 128 --n_heads 1 --e_layers 1 --d_ff 128 --patch_len 8 --k 1 --use_norm 1 --des entrenamiento_inicial --freq d

The checkpoint must exist at:
    ./checkpoints/<setting>/checkpoint.pth

The scaler (saved automatically by exp_term_forecasting.py during training) must
exist at:
    ./checkpoints/<setting>/scaler.pkl

If the scaler file is missing the script will re-fit a StandardScaler on the
training portion (first 70 %) of the CSV as a fallback.
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_setting(args, ii=0):
    """Reproduce the `setting` string used by run.py so we can find the checkpoint."""
    return (
        '{model_id}_{model}_{data}_{features}'
        '_ft{seq_len}_sl{label_len}_ll{pred_len}'
        '_pl{d_model}_dm{n_heads}_nh{e_layers}'
        '_el{d_layers}_dl{d_ff}_df{factor}'
        '_fc{embed}_eb{distil}_dt{des}'
        '_{class_strategy}_{ii}'
    ).format(
        model_id=args.model_id,
        model=args.model,
        data=args.data,
        features=args.features,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=getattr(args, 'd_layers', 1),
        d_ff=args.d_ff,
        factor=getattr(args, 'factor', 3),
        embed=getattr(args, 'embed', 'timeF'),
        distil=getattr(args, 'distil', True),
        des=args.des,
        class_strategy=getattr(args, 'class_strategy', 'projection'),
        ii=ii,
    )


def load_scaler(checkpoint_dir, csv_path, target, features, seq_len):
    """
    Load the pre-saved scaler, or fall back to fitting one on 70 % of the data.

    Parameters
    ----------
    checkpoint_dir : str
        Directory where `scaler.pkl` is expected.
    csv_path : str
        Path to the CSV dataset.
    target : str
        Target column name.
    features : str
        'S', 'M', or 'MS'.
    seq_len : int
        Input window length (used to determine training border).

    Returns
    -------
    scaler : sklearn.preprocessing.StandardScaler
    """
    scaler_path = os.path.join(checkpoint_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        print(f'Loading scaler from {scaler_path}')
        return joblib.load(scaler_path)

    print(f'Scaler not found at {scaler_path}. Fitting a new one on training data (70%).')
    df_raw = pd.read_csv(csv_path)
    # Re-order columns as done in Dataset_Custom
    cols = list(df_raw.columns)
    cols.remove(target)
    cols.remove('date')
    df_raw = df_raw[['date'] + cols + [target]]

    if features in ('M', 'MS'):
        df_data = df_raw.iloc[:, 1:]
    else:  # S
        df_data = df_raw[[target]]

    num_train = int(len(df_raw) * 0.7)
    train_data = df_data.iloc[:num_train].values
    scaler = StandardScaler()
    scaler.fit(train_data)
    return scaler


def get_last_window(csv_path, target, features, seq_len, scaler):
    """
    Return the last `seq_len` rows of the CSV in scaled form, ready for the
    model's forward pass.

    Parameters
    ----------
    csv_path : str
    target : str
    features : str  ('S', 'M', or 'MS')
    seq_len : int
    scaler : StandardScaler

    Returns
    -------
    window : np.ndarray  shape (seq_len, n_vars)
    last_date : pd.Timestamp  last date in the dataset
    """
    df_raw = pd.read_csv(csv_path)
    # Ensure same column order as Dataset_Custom
    cols = list(df_raw.columns)
    cols.remove(target)
    cols.remove('date')
    df_raw = df_raw[['date'] + cols + [target]]

    if features in ('M', 'MS'):
        df_data = df_raw.iloc[:, 1:].values
    else:  # S
        df_data = df_raw[[target]].values

    data_scaled = scaler.transform(df_data)
    window = data_scaled[-seq_len:]  # (seq_len, n_vars)

    last_date = pd.to_datetime(df_raw['date'].iloc[-1])
    return window, last_date


def load_model(checkpoint_dir, args, device):
    """
    Build the DFGCN model from args and load the saved checkpoint weights.

    Parameters
    ----------
    checkpoint_dir : str
    args : argparse.Namespace
    device : torch.device

    Returns
    -------
    model : nn.Module (eval mode)
    """
    # Import the model the same way run.py does
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from modelos import DFGCN as DFGCN_module

    model = DFGCN_module.Model(args).float().to(device)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'Checkpoint not found at {checkpoint_path}. '
            'Train the model first with run.py --is_training 1.'
        )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Model loaded from {checkpoint_path}')
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def predict_future(args):
    """
    Run a single forward pass using the last `seq_len` rows of the CSV and
    return future predictions in real COP/USD scale.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f'Using device: {device}')

    # Build the experiment setting string to locate checkpoints
    setting = build_setting(args)
    checkpoint_dir = os.path.join(args.checkpoints, setting)
    print(f'Checkpoint directory: {checkpoint_dir}')

    # Load scaler
    scaler = load_scaler(
        checkpoint_dir, args.data_path, args.target,
        args.features, args.seq_len,
    )

    # Prepare input window
    window, last_date = get_last_window(
        args.data_path, args.target, args.features, args.seq_len, scaler,
    )
    print(f'Last date in dataset: {last_date.date()}')
    print(f'Input window shape: {window.shape}')

    # Load model
    model = load_model(checkpoint_dir, args, device)

    # Build input tensor  [1, seq_len, n_vars]
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)          # [1, pred_len, n_vars]

    pred_np = pred.squeeze(0).cpu().numpy()   # (pred_len, n_vars)

    # Inverse-transform to real COP/USD scale
    n_vars = pred_np.shape[-1]
    pred_real = scaler.inverse_transform(pred_np.reshape(-1, n_vars)).reshape(pred_np.shape)

    # Generate future business dates
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=args.pred_len,
        freq='B',
    )

    # Build result DataFrame
    if n_vars == 1:
        df_result = pd.DataFrame({
            'date': future_dates,
            'predicted_COP_USD': pred_real[:, 0],
        })
    else:
        # For multivariate output, export all columns; last column is the target
        cols = {f'pred_var_{i}': pred_real[:, i] for i in range(n_vars - 1)}
        cols['predicted_COP_USD'] = pred_real[:, -1]
        df_result = pd.DataFrame({'date': future_dates, **cols})

    # Save to CSV
    output_dir = './resultados_futuro'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{args.model_id}_future_predictions.csv')
    df_result.to_csv(output_path, index=False)
    print(f'\nPredictions saved to: {output_path}')

    # Print summary table
    print('\n=== Future COP/USD Predictions ===')
    print(df_result.to_string(index=False))

    return df_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DFGCN: predict future COP/USD values beyond the last date in the CSV'
    )

    # Identification
    parser.add_argument('--model_id', type=str, required=True,
                        help='Experiment identifier (must match the one used during training)')
    parser.add_argument('--model', type=str, default='DFGCN',
                        help='Model name (default: DFGCN)')
    parser.add_argument('--data', type=str, default='custom',
                        help='Dataset type (default: custom)')
    parser.add_argument('--features', type=str, default='S',
                        help='Feature mode: S=univariate, M=multivariate, MS=multi→uni')
    parser.add_argument('--target', type=str, default='COP_USD',
                        help='Target column name')
    parser.add_argument('--des', type=str, default='test',
                        help='Experiment description (must match training value)')
    parser.add_argument('--freq', type=str, default='d',
                        help='Time frequency (d=daily, b=business days, h=hourly, etc.)')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='Base directory for model checkpoints')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the CSV dataset (e.g., datos/tasa_cop_usd.csv)')

    # Window sizes (must match training)
    parser.add_argument('--seq_len', type=int, default=96,
                        help='Input window length (must match training)')
    parser.add_argument('--label_len', type=int, default=48,
                        help='Decoder initial token length (must match training)')
    parser.add_argument('--pred_len', type=int, default=30,
                        help='Number of future steps to predict')

    # Architecture (must match training)
    parser.add_argument('--enc_in', type=int, default=1,
                        help='Number of input variables (must match training)')
    parser.add_argument('--dec_in', type=int, default=1,
                        help='Number of decoder input variables')
    parser.add_argument('--c_out', type=int, default=1,
                        help='Number of output variables')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model embedding dimension')
    parser.add_argument('--n_heads', type=int, default=1,
                        help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=1,
                        help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=128,
                        help='Feed-forward dimension')
    parser.add_argument('--patch_len', type=int, default=8,
                        help='Temporal patch size (must match training)')
    parser.add_argument('--k', type=int, default=1,
                        help='k-NN neighbors for graph construction (must match training)')
    parser.add_argument('--use_norm', type=int, default=1,
                        help='Use RevIN normalization: 1=yes, 0=no (must match training)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--activation', type=str, default='sigmoid',
                        help='Activation function: sigmoid or relu')
    parser.add_argument('--factor', type=int, default=3,
                        help='Attention factor')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Embedding type')
    parser.add_argument('--distil', action='store_false', default=True,
                        help='Use distillation in encoder')
    parser.add_argument('--output_attention', action='store_true', default=False,
                        help='Output attention weights')
    parser.add_argument('--class_strategy', type=str, default='projection',
                        help='Classification strategy')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='Use GPU if available')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index')

    args = parser.parse_args()
    args.use_gpu = torch.cuda.is_available() and args.use_gpu
    # Alias batch_size (not used for inference but needed by model __init__)
    args.batch_size = 1

    predict_future(args)
