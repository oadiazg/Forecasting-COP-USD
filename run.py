import argparse
import torch
from experiments.exp_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='DFGCN - Predicción de Tasas de Cambio COP/USD')

    # Configuración básica
    parser.add_argument('--is_training', type=int, required=True, default=1,
                        help='Modo: 1 = entrenamiento, 0 = prueba/simulación')
    parser.add_argument('--model_id', type=str, required=True, default='COP_USD',
                        help='Identificador del modelo/experimento')
    parser.add_argument('--model', type=str, required=True, default='DFGCN',
                        help='Nombre del modelo (solo DFGCN disponible)')

    # Carga de datos
    parser.add_argument('--data', type=str, required=True, default='custom',
                        help='Tipo de dataset: custom, ETTh1, ETTh2, ETTm1, ETTm2, PEMS, Solar')
    parser.add_argument('--root_path', type=str, default='./datos/',
                        help='Ruta raíz del archivo de datos')
    parser.add_argument('--data_path', type=str, default='tasa_cop_usd.csv',
                        help='Nombre del archivo CSV de datos')
    parser.add_argument('--features', type=str, default='M',
                        help='Tipo de predicción: M=multivariada, S=univariada, MS=multi→uni')
    parser.add_argument('--target', type=str, default='OT',
                        help='Variable objetivo en tareas S o MS')
    parser.add_argument('--freq', type=str, default='d',
                        help='Frecuencia temporal: s=segundos, t=minutos, h=horas, d=días, '
                             'b=días hábiles, w=semanas, m=meses')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='Directorio donde se guardan los checkpoints del modelo')

    # Tarea de predicción
    parser.add_argument('--seq_len', type=int, default=96,
                        help='Longitud de la ventana de entrada (número de pasos históricos)')
    parser.add_argument('--label_len', type=int, default=48,
                        help='Longitud del token inicial del decodificador')
    parser.add_argument('--pred_len', type=int, default=30,
                        help='Horizonte de predicción (pasos a predecir hacia el futuro)')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly',
                        help='Patrones estacionales para M4')

    # Definición del modelo
    parser.add_argument('--enc_in', type=int, default=7,
                        help='Número de variables de entrada del encoder')
    parser.add_argument('--dec_in', type=int, default=7,
                        help='Número de variables de entrada del decoder')
    parser.add_argument('--c_out', type=int, default=7,
                        help='Número de variables de salida')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Dimensión del modelo (tamaño del embedding interno)')
    parser.add_argument('--n_heads', type=int, default=1,
                        help='Número de cabezas de atención')
    parser.add_argument('--e_layers', type=int, default=1,
                        help='Número de capas del encoder')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='Número de capas del decoder')
    parser.add_argument('--d_ff', type=int, default=128,
                        help='Dimensión de la capa feed-forward')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='Tamaño de ventana para media móvil')
    parser.add_argument('--factor', type=int, default=3,
                        help='Factor de atención')
    parser.add_argument('--distil', action='store_false',
                        help='Si se usa destilación en el encoder',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Tasa de dropout para regularización')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Tipo de embedding temporal: timeF, fixed, learned')
    parser.add_argument('--activation', type=str, default='sigmoid',
                        help='Función de activación: sigmoid, relu')
    parser.add_argument('--output_attention', action='store_true',
                        help='Si se retorna la atención del encoder')

    # Optimización
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Número de workers para el DataLoader')
    parser.add_argument('--itr', type=int, default=1,
                        help='Número de repeticiones del experimento')
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Tamaño del batch de entrenamiento')
    parser.add_argument('--patience', type=int, default=3,
                        help='Paciencia para early stopping')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Tasa de aprendizaje del optimizador')
    parser.add_argument('--des', type=str, default='test',
                        help='Descripción del experimento')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='Función de pérdida: MSE')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='Estrategia de ajuste de learning rate: type1, type2, type3, constant, cosine')
    parser.add_argument('--use_amp', action='store_true',
                        help='Usar precisión mixta automática (AMP)',
                        default=False)
    parser.add_argument('--pct_start', type=float, default=0.3,
                        help='Porcentaje de warmup en OneCycleLR')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='Usar GPU si está disponible')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Índice de la GPU a usar')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='Usar múltiples GPUs',
                        default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='IDs de las GPUs a usar en modo multi-GPU')

    # Parámetros específicos de DFGCN
    parser.add_argument('--use_norm', type=int, default=1,
                        help='Usar normalización reversible RevIN: 1=sí, 0=no')
    parser.add_argument('--exp_name', type=str, required=False, default='None',
                        help='Nombre del experimento: None, partial_train, zero_shot')
    parser.add_argument('--efficient_training', type=bool, default=False,
                        help='Entrenamiento eficiente con selección aleatoria de variables')
    parser.add_argument('--channel_independence', type=bool, default=False,
                        help='Modo channel-independence')
    parser.add_argument('--inverse', action='store_true',
                        help='Invertir la normalización en la salida',
                        default=False)
    parser.add_argument('--class_strategy', type=str, default='projection',
                        help='Estrategia de clasificación: projection, average, cls_token')
    parser.add_argument('--target_root_path', type=str, default='./datos/',
                        help='Ruta raíz para el dataset objetivo (zero-shot)')
    parser.add_argument('--target_data_path', type=str, default='tasa_cop_usd.csv',
                        help='Archivo de datos objetivo (zero-shot)')
    parser.add_argument('--embed_size', type=int, default=128,
                        help='Dimensión de embeddings ocultos')
    parser.add_argument('--k', type=int, default=2,
                        help='Número de vecinos en el grafo de correlación de Pearson (k-NN). '
                             'Must be < enc_in. Use k=1 for univariate (enc_in=1).')
    parser.add_argument('--patch_len', type=int, default=8,
                        help='Longitud del parche temporal para procesamiento por patches')
    parser.add_argument('--report_real_metrics', type=int, default=1,
                        help='Report metrics on real (inverse-transformed) scale in addition to '
                             'normalized scale: 1=yes (default), 0=no')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.exp_name == 'partial_train':
        Exp = Exp_Long_Term_Forecast_Partial
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args)
            print('>>>>>>>inicio entrenamiento: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>prueba/validación: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)
        print('>>>>>>>simulación/prueba: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
