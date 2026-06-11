import subprocess
import re
import torch


def get_gpu_info():
    """获取完整的GPU信息（包含显存）"""
    gpu_info = {
        'vendor': None,  # NVIDIA, AMD, Intel
        'name': None,
        'cuda_version': None,
        'driver_version': None,
        'memory_total_mb': None,  # 总显存 (MB)
        'memory_free_mb': None,  # 可用显存 (MB)
        'memory_used_mb': None  # 已用显存 (MB)
    }

    # 1. 尝试通过 nvidia-smi 获取 NVIDIA GPU 详细信息
    try:
        # 获取显卡名称和驱动版本
        result = subprocess.run(
            'nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used --format=csv,noheader',
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info['vendor'] = 'NVIDIA'
            parts = [p.strip() for p in result.stdout.strip().split(',')]
            gpu_info['name'] = parts[0]
            if len(parts) > 1:
                gpu_info['driver_version'] = parts[1]
            if len(parts) > 2:
                # 解析显存，格式如 "8192 MiB"
                mem_total = re.search(r'(\d+)', parts[2])
                if mem_total:
                    gpu_info['memory_total_mb'] = int(mem_total.group(1))
            if len(parts) > 3:
                mem_free = re.search(r'(\d+)', parts[3])
                if mem_free:
                    gpu_info['memory_free_mb'] = int(mem_free.group(1))
            if len(parts) > 4:
                mem_used = re.search(r'(\d+)', parts[4])
                if mem_used:
                    gpu_info['memory_used_mb'] = int(mem_used.group(1))

            # 获取 CUDA 版本
            cuda_result = subprocess.run(
                'nvidia-smi',
                shell=True, capture_output=True, text=True
            )
            for line in cuda_result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    version_match = re.search(r'CUDA Version:\s*(\d+\.\d+)', line)
                    if version_match:
                        gpu_info['cuda_version'] = version_match.group(1)
                    break
    except:
        pass

    # 2. 如果没找到 NVIDIA，尝试检测 AMD GPU 及其显存
    if not gpu_info['vendor']:
        try:
            result = subprocess.run(
                'powershell "Get-CimInstance Win32_VideoController | Where-Object {$_.Name -like \'*AMD*\' -or $_.Name -like \'*Radeon*\'} | Select-Object Name, AdapterRAM"',
                shell=True, capture_output=True, text=True
            )
            if result.stdout.strip():
                gpu_info['vendor'] = 'AMD'
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    # 解析第二行（数据行）
                    data_parts = lines[1].split()
                    if data_parts:
                        gpu_info['name'] = ' '.join(data_parts[:-1]) if len(data_parts) > 1 else data_parts[0]
                        # 解析显存（AdapterRAM 单位是字节）
                        mem_bytes = data_parts[-1] if data_parts else '0'
                        if mem_bytes.isdigit():
                            mem_mb = int(mem_bytes) / (1024 * 1024)
                            gpu_info['memory_total_mb'] = int(mem_mb)
        except:
            pass

    # 3. 检测 Intel GPU 及其显存
    if not gpu_info['vendor']:
        try:
            result = subprocess.run(
                'powershell "Get-CimInstance Win32_VideoController | Where-Object {$_.Name -like \'*Intel*\'} | Select-Object Name, AdapterRAM"',
                shell=True, capture_output=True, text=True
            )
            if result.stdout.strip():
                gpu_info['vendor'] = 'Intel'
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    data_parts = lines[1].split()
                    if data_parts:
                        gpu_info['name'] = ' '.join(data_parts[:-1]) if len(data_parts) > 1 else data_parts[0]
                        mem_bytes = data_parts[-1] if data_parts else '0'
                        if mem_bytes.isdigit():
                            mem_mb = int(mem_bytes) / (1024 * 1024)
                            gpu_info['memory_total_mb'] = int(mem_mb)
        except:
            pass

    return gpu_info


def format_memory(mb):
    """格式化显存大小显示"""
    if mb is None:
        return "未知"
    if mb >= 1024:
        return f"{mb:.0f} MB ({mb / 1024:.2f} GB)"
    else:
        return f"{mb:.0f} MB"


def get_pytorch_install_command(gpu_info):
    """根据 GPU 信息动态生成 PyTorch 安装命令"""

    commands = []

    # NVIDIA GPU
    if gpu_info['vendor'] == 'NVIDIA':
        cuda_version = gpu_info.get('cuda_version', '')
        total_memory = gpu_info.get('memory_total_mb', 0)

        # 根据显存大小添加提示
        if total_memory and total_memory < 4096:
            commands.append({
                'description': '⚠️ 显存较小 (<4GB)，建议使用轻量级配置',
                'command': '提示：可考虑使用较小的 batch_size 或混合精度训练'
            })

        # 根据 CUDA 版本选择
        if cuda_version:
            cuda_major = float(cuda_version.split('.')[0])

            if cuda_major >= 12:
                commands.append({
                    'description': 'CUDA 12.x (推荐，性能最佳)',
                    'command': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
                })
                commands.append({
                    'description': 'CUDA 11.8 (兼容性更好)',
                    'command': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
                })
            elif cuda_major >= 11:
                commands.append({
                    'description': f'CUDA {cuda_version} (匹配您的驱动)',
                    'command': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
                })
            else:
                commands.append({
                    'description': 'CPU 版本 (您的 CUDA 版本较旧)',
                    'command': 'pip install torch torchvision torchaudio'
                })
        else:
            # 无法检测 CUDA 版本时，根据 GPU 型号推断
            gpu_name = gpu_info.get('name', '').lower()
            if 'rtx 40' in gpu_name or 'rtx 50' in gpu_name:
                commands.append({
                    'description': '推荐 (RTX 40/50 系列显卡)',
                    'command': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
                })
            elif 'rtx 30' in gpu_name or 'rtx 20' in gpu_name:
                commands.append({
                    'description': '推荐 (RTX 30/20 系列显卡)',
                    'command': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
                })
            else:
                commands.append({
                    'description': '通用 NVIDIA GPU',
                    'command': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
                })

    # AMD GPU (ROCm)
    elif gpu_info['vendor'] == 'AMD':
        commands.append({
            'description': 'AMD GPU (ROCm 5.6)',
            'command': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6'
        })
        commands.append({
            'description': 'AMD GPU (CPU 版本备用)',
            'command': 'pip install torch torchvision torchaudio'
        })

    # Intel GPU
    elif gpu_info['vendor'] == 'Intel':
        commands.append({
            'description': 'Intel GPU (实验性支持)',
            'command': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu'
        })
        commands.append({
            'description': 'Intel GPU (CPU 版本)',
            'command': 'pip install torch torchvision torchaudio'
        })

    # 未检测到 GPU
    else:
        commands.append({
            'description': 'CPU 版本 (未检测到兼容 GPU)',
            'command': 'pip install torch torchvision torchaudio'
        })
        commands.append({
            'description': '如需 GPU 支持，请访问官网选择',
            'command': 'https://pytorch.org/get-started/locally/'
        })

    return commands


def display_install_guide():
    """显示完整的安装指南（包含显存信息）"""
    print("=" * 70)
    print("GPU 检测与 PyTorch 安装建议")
    print("=" * 70)

    # 1. 获取 GPU 信息
    print("\n【检测到的 GPU 信息】")
    gpu_info = get_gpu_info()

    if gpu_info['vendor']:
        print(f"厂商: {gpu_info['vendor']}")
        print(f"型号: {gpu_info['name']}")

        # 显示显存信息
        if gpu_info['memory_total_mb']:
            print(f"总显存: {format_memory(gpu_info['memory_total_mb'])}")
            if gpu_info['memory_free_mb']:
                print(f"可用显存: {format_memory(gpu_info['memory_free_mb'])}")
            if gpu_info['memory_used_mb']:
                print(f"已用显存: {format_memory(gpu_info['memory_used_mb'])}")
                usage_rate = (gpu_info['memory_used_mb'] / gpu_info['memory_total_mb']) * 100
                print(f"显存使用率: {usage_rate:.1f}%")
        else:
            print("显存大小: 无法获取")

        if gpu_info['cuda_version']:
            print(f"CUDA 版本: {gpu_info['cuda_version']}")
        if gpu_info['driver_version']:
            print(f"驱动版本: {gpu_info['driver_version']}")

        # 根据显存大小给出建议
        if gpu_info['memory_total_mb']:
            mem_gb = gpu_info['memory_total_mb'] / 1024
            if mem_gb >= 16:
                print("\n💡 提示: 大显存 GPU，适合训练大型模型")
            elif mem_gb >= 8:
                print("\n💡 提示: 中等显存，适合训练中小型模型")
            elif mem_gb >= 4:
                print("\n💡 提示: 入门级显存，建议使用较小 batch_size")
            else:
                print("\n⚠️ 警告: 显存较小，建议使用 CPU 或云 GPU 服务")
    else:
        print("未检测到独立 GPU，将使用 CPU 版本")

    # 2. 生成安装命令
    print("\n【为您生成的 PyTorch 安装命令】")
    commands = get_pytorch_install_command(gpu_info)

    for i, cmd_info in enumerate(commands, 1):
        print(f"\n{i}. {cmd_info['description']}:")
        print(f"   {cmd_info['command']}")

    # 3. 额外建议
    print("\n" + "=" * 70)
    print("【安装前注意事项】")
    print("1. 建议先卸载旧版本: pip uninstall torch torchvision torchaudio")
    print("2. 创建虚拟环境避免冲突: python -m venv pytorch_env")
    print("3. 激活虚拟环境后再执行上述安装命令")

    # 根据显存大小添加具体建议
    if gpu_info['memory_total_mb'] and gpu_info['memory_total_mb'] < 8192:
        print("\n【针对您显存大小的特别建议】")
        print("• 使用 batch_size = 16-32 开始尝试")
        print("• 启用混合精度训练 (torch.cuda.amp)")
        print("• 考虑使用梯度累积技术")

    print("4. 安装后验证: python -c \"import torch; print(torch.cuda.is_available())\"")
    print("=" * 70)


# 同时检测 PyTorch 当前状态
def check_current_pytorch():
    print("\n【当前 PyTorch 状态】")
    print(f"PyTorch 版本: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA 可用: True")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"  GPU {i}: {name}，总显存: {mem_gb:.2f} GB")
    else:
        print("CUDA 可用: False")
        print("⚠️ 当前是 CPU 版本，如需使用 GPU 请按上述建议重新安装")


# 主程序
if __name__ == "__main__":
    display_install_guide()
    check_current_pytorch()