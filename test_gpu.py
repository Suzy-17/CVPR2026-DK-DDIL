import torch
import time
import numpy as np

def test_gpu(gpu_id):
    """执行单张GPU的完整测试流程"""
    torch.cuda.empty_cache()
    device = torch.device(f'cuda:{gpu_id}')
    
    # 阶段1：设备基本信息检测
    try:
        gpu_name = torch.cuda.get_device_name(gpu_id)
        capability = torch.cuda.get_device_capability(gpu_id)
        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        print(f"\n[✅] GPU {gpu_id} 基本信息:")
        print(f" - 设备名称: {gpu_name}")
        print(f" - 计算能力: {capability[0]}.{capability[1]}")
        print(f" - 显存容量: {total_mem:.2f} GB")
    except Exception as e:
        print(f"[❌] GPU {gpu_id} 基本信息获取失败: {e}")
        return False

    # 阶段2：计算功能测试
    try:
        start_time = time.time()
        # 创建大矩阵（触发显存压力测试）
        x = torch.randn(10000, 10000, device=device)
        y = torch.randn(10000, 10000, device=device)
        # 执行矩阵乘法（高负载计算）
        z = torch.matmul(x, y)
        # 验证结果有效性
        if not torch.allclose(z.mean(), torch.tensor(0.0, device=device), atol=1e-1):
            raise ValueError("计算结果异常")
        compute_time = time.time() - start_time
        print(f"[✅] 计算测试通过 | 耗时: {compute_time:.2f}s | 峰值显存: {torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB")
    except Exception as e:
        print(f"[❌] GPU {gpu_id} 计算测试失败: {e}")
        return False

    # 阶段3：显存稳定性测试
    try:
        # 分阶段申请显存（检测碎片处理能力）
        block_list = []
        for _ in range(10):
            block = torch.empty(int(200e6), dtype=torch.uint8, device=device)
            block_list.append(block)
        # 显存释放验证
        del block_list
        torch.cuda.empty_cache()
        if torch.cuda.memory_allocated(device) > 1024**2:
            raise RuntimeError("显存释放异常")
        print(f"[✅] 显存稳定性测试通过")
    except Exception as e:
        print(f"[❌] GPU {gpu_id} 显存测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("PyTorch GPU 健康检测工具")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print("="*60)

    if not torch.cuda.is_available():
        print("未检测到可用GPU设备，请检查驱动安装！")
        exit(1)

    total_gpus = torch.cuda.device_count()
    print(f"检测到 {total_gpus} 张GPU，开始全面测试...\n")
    
    # 执行所有GPU测试
    results = {}
    for gpu_id in range(total_gpus):
        results[gpu_id] = test_gpu(gpu_id)
    
    # 生成测试报告
    print("\n" + "="*60)
    print("GPU健康测试报告：")
    for gpu_id, success in results.items():
        status = "通过" if success else "失败"
        print(f"GPU {gpu_id}: [{status}] - {torch.cuda.get_device_name(gpu_id)}")
    
    print(f"\n总通过率: {sum(results.values())}/{total_gpus}")
    print("="*60)