import argparse
from src.utils import set_seed, check_dir

def main(args):
    set_seed(42)
    # 保持原有的目录检查
    check_dir(f"{args.out_dir}/figures")
    check_dir(f"{args.out_dir}/models")
    check_dir(f"{args.out_dir}/preds")
    
    scaler_path = f"{args.out_dir}/models/scaler_{args.mode}.pkl"
    
    if args.mode == 'short':
        print(f"\n>>> Launching Short-Term Forecasting Pipeline | Model: {args.model} <<<")
        from src.data_handler_short import get_data
        from src.trainer_short import Trainer
        
        #args.seq_len = 168 # 7 days
        #args.pred_len = 24 # 24 hours
        
        # 加载数据（保持原有可视化）
        train_loader, val_loader, test_loader, scaler = get_data(
            args.file_path, args.seq_len, args.pred_len, args.batch_size,
            save_dir=f"{args.out_dir}/figures", scaler_save_path=scaler_path
        )
        
        batch_data = next(iter(train_loader))
        input_dim = batch_data[0].shape[2]
        print(f"Input Features: {input_dim}")
        
        # =======================================================
        # 模型适配逻辑：根据参数加载不同模型
        # =======================================================
        if args.model == 'tqnet_enhanced':
            from src.models.tqnet_enhanced import ShortTermTQNet
            model = ShortTermTQNet(input_dim=input_dim, 
                                   seq_len=args.seq_len, 
                                   pred_len=args.pred_len, 
                                   d_model=args.d_model, 
                                   n_heads=args.n_heads, 
                                   kernel_size=args.kernel_size,
                                   tau=args.tau, 
                                   use_revin=not args.no_revin, 
                                   use_cnn=not args.no_cnn,
                                   use_attention=not args.no_attention, 
                                   use_trend=not args.no_trend)
            
        elif args.model == 'dlinear':
            from src.models.dlinear import DLinear_Model
            model = DLinear_Model(input_dim, args.seq_len, args.pred_len)
            
        elif args.model == 'tqnet_vanilla':
            from src.models.tqnet_vanilla import TQNet_Vanilla
            model = TQNet_Vanilla(input_dim, args.seq_len, args.pred_len)
            
        elif args.model == 'lstm':
            from src.models.lstm import LSTM_Adapter # 需要你新建此文件
            model = LSTM_Adapter(input_dim, args.seq_len, args.pred_len, hidden_dim=128)
            
        elif args.model == 'patchtst':
            # 【新增】集成 PatchTST
            from src.models.patchtst_adapter import PatchTST_Adapter
            model = PatchTST_Adapter(
                input_dim=input_dim, 
                seq_len=args.seq_len, 
                pred_len=args.pred_len,
                d_model=args.d_model,
            )
        elif args.model in ['iTransformer', 'Autoformer', 'Crossformer', 'FEDformer']:
            # 集成来自 Time-Series-Library 的 SOTA 模型
            from src.models.tslib_adapter import TSLib_Adapter
            model = TSLib_Adapter(
                model_name=args.model,
                args=args,
                input_dim=input_dim
            )    
            # =======================================================

        # 保持原有的 Trainer 调用，包括可视化和结果保存
        trainer = Trainer(model, train_loader, val_loader, test_loader, scaler, args)
        trainer.train()
        trainer.test()
    
    # ==========================================
    # 模式 2：长时预测 Pipeline
    # ==========================================
    elif args.mode == 'long':
        print(f"\n>>> Launching Long-Term Forecasting Pipeline | Model: {args.model} <<<")
        # 共用短时的数据处理框架，但是调用 data_handler_long 中的封装
        from src.data_handler_long import get_data
        from src.trainer_long import TrainerLong
        
        train_loader, val_loader, test_loader, scaler, raw_dates = get_data(
            args.file_path, args.seq_len, args.pred_len, args.batch_size,
            save_dir=f"{args.out_dir}/figures", scaler_save_path=scaler_path
        )
        
        batch_data = next(iter(train_loader))
        input_dim = batch_data[0].shape[2]
        print(f"Long-Term Input Features: {input_dim}")
        
        # 加载长时创新模型
        if args.model == 'tqnet_dual':
            from src.models_long import LongTermDualTQNet
            model = LongTermDualTQNet(input_dim, args.seq_len, args.pred_len, d_model=args.d_model)
        elif args.model == 'dlinear_long':
            from src.models_long import LongTermDLinear
            model = LongTermDLinear(input_dim, args.seq_len, args.pred_len, d_model=args.d_model)
        else:
            raise ValueError(f"Model {args.model} not supported in long mode.")
            
        trainer = TrainerLong(model, train_loader, val_loader, test_loader, scaler, raw_dates, args)
        trainer.train()
        trainer.test()
    
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./data/all_energy_clean_modified.csv')
    parser.add_argument('--out_dir', type=str, default='./output')
    parser.add_argument('--mode', type=str, default='short', required=True)
    
    # 新增模型选择参数
    parser.add_argument('--model', type=str, default='tqnet_enhanced', 
                        choices=['tqnet_enhanced', 'dlinear', 'tqnet_vanilla', 'lstm', 'patchtst', 
                            'iTransformer', 'Autoformer', 'Crossformer', 'FEDformer','tqnet_dual','dlinear_long'])
    #增加消融参数
    parser.add_argument('--no_revin', action='store_true')
    parser.add_argument('--no_cnn', action='store_true')
    parser.add_argument('--no_attention', action='store_true')
    parser.add_argument('--no_weighted_loss', action='store_true')
    parser.add_argument('--no_trend', action='store_true', help='Ablation: Disable Linear Trend branch')
    # 新增超参数分析参数
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--tau', type=float, default=0.5, help='temperature factor for attention')
    parser.add_argument('--loss_weight', type=float, default=5.0, help='weight for critical hours')
    
    parser.add_argument('--epochs', type=int, default=400) 
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--seq_len', type=int, default=24, help='输入历史窗口长度')
    parser.add_argument('--pred_len', type=int, default=720, help='预测未来长度')
    parser.add_argument('--kernel_size', type=int, default=3, help='CNN卷积核大小(必须是奇数)')
    args = parser.parse_args()
    main(args)