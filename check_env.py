#!/usr/bin/env python3
"""
환경 확인 및 안전한 GPU 테스트 스크립트
NVML/CUDA 오류 해결을 위한 진단 도구
"""

import os
import sys
import subprocess

def check_environment():
    """현재 환경 정보 확인"""
    print("🔍 환경 정보 확인")
    print("=" * 50)
    
    # 기본 시스템 정보
    print(f"Python 버전: {sys.version}")
    print(f"OS: {os.uname().sysname} {os.uname().release}")
    
    # PyTorch 정보
    try:
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                try:
                    props = torch.cuda.get_device_properties(i)
                    print(f"    메모리: {props.total_memory / 1024**3:.1f} GB")
                except Exception as e:
                    print(f"    메모리 정보 실패: {e}")
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다")
    
    # NVIDIA 드라이버 정보
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi 사용 가능")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"드라이버 버전: {line.split('Driver Version: ')[1].split()[0]}")
                    break
        else:
            print("❌ nvidia-smi 실행 실패")
    except FileNotFoundError:
        print("❌ nvidia-smi를 찾을 수 없습니다")
    
    # 환경 변수 확인
    print("\n🔧 환경 변수 확인")
    print("=" * 50)
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'PYTORCH_CUDA_ALLOC_CONF', 
        'CUDA_LAUNCH_BLOCKING',
        'NVIDIA_DISABLE_REQUIRE'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, '설정되지 않음')
        print(f"{var}: {value}")

def set_safe_environment():
    """안전한 환경 변수 설정"""
    print("\n⚙️  안전한 환경 변수 설정")
    print("=" * 50)
    
    # GPU 메모리 관리 최적화
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    print("✅ PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256")
    print("✅ CUDA_LAUNCH_BLOCKING=1")
    
    # NVML 문제 해결 시도
    try:
        import pynvml
        pynvml.nvmlInit()
        print("✅ NVML 초기화 성공")
    except Exception as e:
        print(f"⚠️  NVML 초기화 실패: {e}")
        os.environ['NVIDIA_DISABLE_REQUIRE'] = '1'
        print("✅ NVIDIA_DISABLE_REQUIRE=1 설정")

def run_safe_gpu_test():
    """안전한 GPU 테스트 실행"""
    print("\n🧪 안전한 GPU 테스트 실행")
    print("=" * 50)
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA를 사용할 수 없습니다")
            return
        
        device_count = torch.cuda.device_count()
        print(f"🎯 {device_count}개 GPU에서 안전 테스트 시작")
        
        for i in range(device_count):
            print(f"\n🚀 GPU {i} 테스트 중...")
            device = torch.device(f'cuda:{i}')
            
            try:
                # 작은 텐서로 테스트
                x = torch.randn(100, 100, device=device)
                y = torch.randn(100, 100, device=device)
                z = torch.matmul(x, y)
                
                # 메모리 정보
                allocated = torch.cuda.memory_allocated(i) / 1024**2
                reserved = torch.cuda.memory_reserved(i) / 1024**2
                
                print(f"  ✅ 기본 연산 성공")
                print(f"  📊 메모리: {allocated:.1f}MB 할당, {reserved:.1f}MB 예약")
                
                # 메모리 정리
                del x, y, z
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ❌ GPU {i} 테스트 실패: {e}")
        
        print(f"\n✅ 안전 테스트 완료")
        
    except ImportError:
        print("❌ PyTorch를 import할 수 없습니다")
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

def main():
    """메인 함수"""
    print("🛠️  GPU 환경 진단 및 안전 테스트")
    print("=" * 60)
    
    # 1. 환경 확인
    check_environment()
    
    # 2. 안전한 환경 설정
    set_safe_environment()
    
    # 3. 안전한 GPU 테스트
    run_safe_gpu_test()
    
    # 4. 권장 명령어 제시
    print("\n💡 권장 해결책")
    print("=" * 50)
    print("1. 환경 변수 설정:")
    print("   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256")
    print("   export CUDA_LAUNCH_BLOCKING=1")
    print("")
    print("2. 안전한 GPU 테스트 실행:")
    print("   python pytorch_tests/gpu_utilization_test.py --duration 300")
    print("")
    print("3. 배치 크기 제한 테스트:")
    print("   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \\")
    print("   python run_tests.py pytorch gpu_utilization --duration 600")
    print("")
    print("4. NVML 문제 시:")
    print("   export NVIDIA_DISABLE_REQUIRE=1")
    print("   python pytorch_tests/gpu_utilization_test.py --duration 300")

if __name__ == "__main__":
    main() 