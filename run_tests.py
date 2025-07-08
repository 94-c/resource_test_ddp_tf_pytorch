#!/usr/bin/env python3
"""
리소스 테스트 실행 스크립트
PyTorch와 TensorFlow를 사용한 CPU, 메모리, GPU 메모리, GPU 사용률 테스트를 실행합니다.
"""

import os
import sys
import subprocess
import argparse
from typing import List, Dict, Optional

def get_available_tests() -> Dict[str, Dict[str, str]]:
    """사용 가능한 테스트 목록 반환"""
    return {
        'pytorch': {
            'cpu': 'pytorch_tests/cpu_test.py',
            'memory': 'pytorch_tests/memory_test.py', 
            'gpu_memory': 'pytorch_tests/gpu_memory_test.py',
            'gpu_utilization': 'pytorch_tests/gpu_utilization_test.py',
            'ddp_training': 'pytorch_tests/ddp_training_test.py'
        },
        'tensorflow': {
            'cpu': 'tensorflow_tests/cpu_test.py',
            'memory': 'tensorflow_tests/memory_test.py',
            'gpu_memory': 'tensorflow_tests/gpu_memory_test.py',
            'gpu_utilization': 'tensorflow_tests/gpu_utilization_test.py'
        }
    }

def check_file_exists(file_path: str) -> bool:
    """파일 존재 여부 확인"""
    return os.path.isfile(file_path)

def run_test(test_path: str, additional_args: Optional[List[str]] = None) -> int:
    """테스트 실행"""
    if not check_file_exists(test_path):
        print(f"❌ 테스트 파일을 찾을 수 없습니다: {test_path}")
        return 1
    
    cmd = [sys.executable, test_path]
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"🚀 실행 중: {test_path}")
    print(f"   명령어: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  사용자에 의해 테스트가 중단되었습니다.")
        return 130
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        return 1

def print_test_menu():
    """테스트 메뉴 출력"""
    tests = get_available_tests()
    
    print("\n🧪 사용 가능한 리소스 테스트:")
    print("=" * 50)
    
    for framework, test_types in tests.items():
        print(f"\n📁 {framework.upper()}:")
        for test_type, test_path in test_types.items():
            status = "✅" if check_file_exists(test_path) else "❌"
            print(f"  {status} {test_type:<15} - {test_path}")
    
    print("\n💡 사용법 예시:")
    print("  python run_tests.py pytorch cpu --duration 60")
    print("  python run_tests.py tensorflow memory --num-tensors 20")
    print("  python run_tests.py pytorch gpu_utilization --duration 600  # DCGM 메트릭 대응")
    print("  python run_tests.py pytorch ddp_training --epochs 10 --batch-size 64")
    print("  python run_tests.py --list  # 전체 테스트 목록 보기")
    print("  python run_tests.py --intensive-gpu  # 집약적 GPU 테스트 (DCGM 메트릭용)")

def main():
    parser = argparse.ArgumentParser(
        description='리소스 테스트 실행기 - PyTorch/TensorFlow CPU, 메모리, GPU 테스트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  %(prog)s pytorch cpu --duration 120
  %(prog)s tensorflow memory --num-tensors 30
  %(prog)s pytorch gpu_memory --tensor-size 2000
  %(prog)s tensorflow gpu_utilization --duration 600  # DCGM 메트릭 대응
  %(prog)s pytorch ddp_training --epochs 50 --batch-size 32
  %(prog)s --list
  %(prog)s --intensive-gpu  # 집약적 GPU 테스트 (DCGM 메트릭용)

테스트 타입:
  cpu              - CPU 집약적 테스트 (행렬 연산, 병렬 처리)
  memory           - 메모리 집약적 테스트 (대용량 텐서 할당)
  gpu_memory       - GPU 메모리 집약적 테스트 (GPU 텐서 할당)
  gpu_utilization  - GPU 사용률 집약적 테스트 (계산 집약적 연산)
  ddp_training     - 분산 학습 테스트 (DDP 기반 멀티 GPU 학습)

특수 옵션:
  --intensive-gpu  - DCGM 메트릭에 나타나도록 하는 집약적 GPU 테스트
        """
    )
    
    parser.add_argument('framework', 
                       choices=['pytorch', 'tensorflow'], 
                       nargs='?',
                       help='사용할 프레임워크')
    
    parser.add_argument('test_type', 
                       choices=['cpu', 'memory', 'gpu_memory', 'gpu_utilization', 'ddp_training'],
                       nargs='?',
                       help='실행할 테스트 타입')
    
    parser.add_argument('--list', '-l', action='store_true',
                       help='사용 가능한 테스트 목록 출력')
    
    parser.add_argument('--all-pytorch', action='store_true',
                       help='모든 PyTorch 테스트 순차 실행')
    
    parser.add_argument('--all-tensorflow', action='store_true',
                       help='모든 TensorFlow 테스트 순차 실행')
    
    parser.add_argument('--intensive-gpu', action='store_true',
                       help='집약적 GPU 테스트 (DCGM 메트릭에 나타나도록 10분 이상 실행)')
    
    parser.add_argument('--duration', type=int, default=60,
                       help='테스트 지속 시간 (초, 기본값: 60)')
    
    # 개별 테스트에 전달할 추가 인자들을 캡처
    parser.add_argument('additional_args', nargs='*',
                       help='테스트 스크립트에 전달할 추가 인자')
    
    args = parser.parse_args()
    
    # 테스트 목록 출력
    if args.list:
        print_test_menu()
        return 0
    
    tests = get_available_tests()
    
    # 집약적 GPU 테스트 실행
    if args.intensive_gpu:
        print("🔥 집약적 GPU 테스트를 실행합니다...")
        print("   DCGM_FI_PROF_GR_ENGINE_ACTIVE 메트릭에 나타나도록 10분 이상 실행됩니다.")
        print("   Ctrl+C로 중단할 수 있습니다.")
        
        # 기본 지속 시간을 10분으로 설정
        intensive_duration = max(600, args.duration)
        additional_args = ['--duration', str(intensive_duration)] + args.additional_args
        
        failed_tests = []
        
        # PyTorch GPU 집약적 테스트
        print(f"\n{'='*60}")
        print("PyTorch GPU 집약적 테스트 시작")
        print(f"{'='*60}")
        
        result = run_test(tests['pytorch']['gpu_utilization'], additional_args)
        if result != 0:
            failed_tests.append("pytorch gpu_utilization")
        
        print(f"\nPyTorch GPU 집약적 테스트 {'완료' if result == 0 else '실패'}")
        
        # TensorFlow GPU 집약적 테스트
        print(f"\n{'='*60}")
        print("TensorFlow GPU 집약적 테스트 시작")
        print(f"{'='*60}")
        
        result = run_test(tests['tensorflow']['gpu_utilization'], additional_args)
        if result != 0:
            failed_tests.append("tensorflow gpu_utilization")
        
        print(f"\nTensorFlow GPU 집약적 테스트 {'완료' if result == 0 else '실패'}")
        
        if failed_tests:
            print(f"\n❌ 실패한 테스트: {', '.join(failed_tests)}")
            return 1
        else:
            print(f"\n✅ 모든 집약적 GPU 테스트가 성공적으로 완료되었습니다!")
            return 0

    # 모든 PyTorch 테스트 실행
    if args.all_pytorch:
        print("🔥 모든 PyTorch 테스트를 순차적으로 실행합니다...")
        failed_tests = []
        
        for test_type, test_path in tests['pytorch'].items():
            print(f"\n{'='*60}")
            print(f"PyTorch {test_type.upper()} 테스트 시작")
            print(f"{'='*60}")
            
            # DDP 테스트는 특별한 인자 처리
            if test_type == 'ddp_training':
                additional_args = ['--epochs', '5', '--batch-size', '16'] + args.additional_args
            else:
                additional_args = ['--duration', str(args.duration)] + args.additional_args
            
            result = run_test(test_path, additional_args)
            
            if result != 0:
                failed_tests.append(f"pytorch {test_type}")
            
            print(f"\nPyTorch {test_type} 테스트 {'완료' if result == 0 else '실패'}")
        
        if failed_tests:
            print(f"\n❌ 실패한 테스트: {', '.join(failed_tests)}")
            return 1
        else:
            print(f"\n✅ 모든 PyTorch 테스트가 성공적으로 완료되었습니다!")
            return 0
    
    # 모든 TensorFlow 테스트 실행
    if args.all_tensorflow:
        print("🔥 모든 TensorFlow 테스트를 순차적으로 실행합니다...")
        failed_tests = []
        
        for test_type, test_path in tests['tensorflow'].items():
            print(f"\n{'='*60}")
            print(f"TensorFlow {test_type.upper()} 테스트 시작")
            print(f"{'='*60}")
            
            additional_args = ['--duration', str(args.duration)] + args.additional_args
            result = run_test(test_path, additional_args)
            
            if result != 0:
                failed_tests.append(f"tensorflow {test_type}")
            
            print(f"\nTensorFlow {test_type} 테스트 {'완료' if result == 0 else '실패'}")
        
        if failed_tests:
            print(f"\n❌ 실패한 테스트: {', '.join(failed_tests)}")
            return 1
        else:
            print(f"\n✅ 모든 TensorFlow 테스트가 성공적으로 완료되었습니다!")
            return 0
    
    # 개별 테스트 실행
    if not args.framework or not args.test_type:
        print("❌ 프레임워크와 테스트 타입을 지정해주세요.")
        print_test_menu()
        return 1
    
    test_path = tests[args.framework][args.test_type]
    
    # 추가 인자 구성 - DDP 테스트는 다른 기본값 사용
    if args.test_type == 'ddp_training':
        additional_args = []
        if args.additional_args:
            additional_args.extend(args.additional_args)
    else:
        additional_args = ['--duration', str(args.duration)]
        if args.additional_args:
            additional_args.extend(args.additional_args)
    
    # 테스트 실행
    return run_test(test_path, additional_args)

if __name__ == "__main__":
    print("🧪 리소스 테스트 실행기")
    print("PyTorch와 TensorFlow를 사용한 CPU, 메모리, GPU 리소스 테스트")
    print("=" * 60)
    
    exit_code = main()
    
    if exit_code == 0:
        print("\n✅ 테스트가 성공적으로 완료되었습니다!")
    else:
        print(f"\n❌ 테스트가 실패했습니다. (종료 코드: {exit_code})")
    
    sys.exit(exit_code) 