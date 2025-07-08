"""
PyTorch CPU 집약적 테스트
이 테스트는 CPU 사용률을 최대화하여 CPU 리소스 모니터링을 테스트합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import numpy as np
import time
import argparse
import sys
import os

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.resource_monitor import ResourceMonitor, print_system_info


def cpu_intensive_matrix_operations(size=2000, iterations=100):
    """CPU 집약적 행렬 연산"""
    print(f"Starting CPU intensive matrix operations (size: {size}x{size}, iterations: {iterations})")
    
    # CPU에서만 동작하도록 설정
    torch.set_num_threads(mp.cpu_count())
    
    results = []
    for i in range(iterations):
        # 랜덤 행렬 생성
        a = torch.randn(size, size, dtype=torch.float32)
        b = torch.randn(size, size, dtype=torch.float32)
        
        # 다양한 CPU 집약적 연산
        # 1. 행렬 곱셈
        c = torch.matmul(a, b)
        
        # 2. 고유값 분해 (매우 CPU 집약적)
        if i % 10 == 0:  # 10번마다 한 번씩 실행
            try:
                eigenvalues, eigenvectors = torch.linalg.eig(c[:500, :500])  # 크기 줄여서 실행
                results.append(eigenvalues.real.mean().item())
            except:
                pass
        
        # 3. SVD 분해
        if i % 5 == 0:  # 5번마다 한 번씩 실행
            try:
                u, s, v = torch.linalg.svd(c[:200, :200])
                results.append(s.mean().item())
            except:
                pass
        
        # 4. 역행렬 계산
        try:
            c_inv = torch.linalg.inv(c + torch.eye(size) * 1e-5)  # 수치적 안정성을 위한 regularization
            results.append(c_inv.trace().item())
        except:
            pass
        
        # 5. 복잡한 수학 연산
        d = torch.sin(c) + torch.cos(c) + torch.exp(torch.clamp(c, -5, 5))
        results.append(d.mean().item())
        
        if i % 10 == 0:
            print(f"  Iteration {i}/{iterations} completed")
    
    return np.mean(results)


def cpu_parallel_processing(num_processes=None, work_per_process=50):
    """멀티프로세싱을 사용한 CPU 병렬 처리"""
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Starting CPU parallel processing ({num_processes} processes)")
    
    def worker_function(worker_id):
        """각 프로세스에서 실행할 작업"""
        results = []
        for i in range(work_per_process):
            # CPU 집약적 연산
            size = 1000
            a = torch.randn(size, size)
            b = torch.randn(size, size)
            
            # 행렬 연산
            c = torch.matmul(a, b)
            d = torch.matmul(c, a.t())
            
            # 통계 계산
            result = {
                'mean': d.mean().item(),
                'std': d.std().item(),
                'max': d.max().item(),
                'min': d.min().item()
            }
            results.append(result)
        
        return f"Worker {worker_id} completed {len(results)} operations"
    
    # 멀티프로세싱 실행
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(worker_function, range(num_processes))
    
    return results


def neural_network_cpu_training(epochs=50, batch_size=1024):
    """CPU에서 신경망 학습"""
    print(f"Starting neural network training on CPU (epochs: {epochs}, batch_size: {batch_size})")
    
    # CPU에서만 동작하도록 강제
    device = torch.device('cpu')
    
    # 복잡한 신경망 모델 정의
    class ComplexCPUModel(nn.Module):
        def __init__(self):
            super(ComplexCPUModel, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(784, 2048),
                nn.ReLU(),
                nn.BatchNorm1d(2048),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = ComplexCPUModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 가짜 데이터 생성
    def generate_batch():
        x = torch.randn(batch_size, 784).to(device)
        y = torch.randint(0, 10, (batch_size,)).to(device)
        return x, y
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 10  # 에포크당 배치 수
        
        for batch_idx in range(num_batches):
            x, y = generate_batch()
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    return losses


def fibonacci_cpu_intensive(n=40):
    """CPU 집약적 피보나치 계산"""
    print(f"Starting CPU intensive Fibonacci calculation (n={n})")
    
    def fibonacci_recursive(n):
        if n <= 1:
            return n
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
    
    # 여러 개의 피보나치 계산을 병렬로 실행
    num_workers = mp.cpu_count()
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(fibonacci_recursive, [n-i for i in range(num_workers)])
    
    return results


def main():
    parser = argparse.ArgumentParser(description='PyTorch CPU 집약적 테스트')
    parser.add_argument('--duration', type=int, default=60, help='테스트 지속 시간 (초)')
    parser.add_argument('--matrix-size', type=int, default=1500, help='행렬 크기')
    parser.add_argument('--iterations', type=int, default=50, help='반복 횟수')
    parser.add_argument('--skip-parallel', action='store_true', help='병렬 처리 테스트 건너뛰기')
    parser.add_argument('--skip-training', action='store_true', help='신경망 학습 테스트 건너뛰기')
    parser.add_argument('--skip-fibonacci', action='store_true', help='피보나치 테스트 건너뛰기')
    
    args = parser.parse_args()
    
    # 시스템 정보 출력
    print_system_info()
    
    # 리소스 모니터링 시작
    monitor = ResourceMonitor(interval=0.5)
    monitor.start_monitoring()
    
    print(f"\n🔥 PyTorch CPU 집약적 테스트 시작 (지속 시간: {args.duration}초)")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 테스트 실행
        test_results = {}
        
        # 1. 행렬 연산 테스트
        print("\n1. 행렬 연산 테스트")
        result = cpu_intensive_matrix_operations(size=args.matrix_size, iterations=args.iterations)
        test_results['matrix_operations'] = result
        
        # 2. 병렬 처리 테스트
        if not args.skip_parallel:
            print("\n2. 병렬 처리 테스트")
            result = cpu_parallel_processing()
            test_results['parallel_processing'] = result
        
        # 3. 신경망 학습 테스트
        if not args.skip_training:
            print("\n3. 신경망 학습 테스트")
            result = neural_network_cpu_training(epochs=30)
            test_results['neural_network'] = result
        
        # 4. 피보나치 테스트
        if not args.skip_fibonacci:
            print("\n4. 피보나치 테스트")
            result = fibonacci_cpu_intensive(n=35)
            test_results['fibonacci'] = result
        
        # 남은 시간 동안 추가 행렬 연산 수행
        elapsed_time = time.time() - start_time
        remaining_time = args.duration - elapsed_time
        
        if remaining_time > 5:
            print(f"\n5. 추가 행렬 연산 ({remaining_time:.1f}초 동안)")
            extra_iterations = max(10, int(remaining_time / 2))
            result = cpu_intensive_matrix_operations(size=1000, iterations=extra_iterations)
            test_results['extra_operations'] = result
        
    except KeyboardInterrupt:
        print("\n테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n테스트 중 오류 발생: {e}")
    finally:
        # 리소스 모니터링 종료
        monitor.stop_monitoring()
        monitor.print_summary()
        
        # 최종 리소스 사용량 확인
        final_usage = monitor.get_current_usage()
        print(f"\n최종 리소스 사용량:")
        print(f"CPU: {final_usage.get('cpu_percent', 0):.1f}%")
        print(f"Memory: {final_usage.get('memory_percent', 0):.1f}%")
        
        print("\n🎯 PyTorch CPU 집약적 테스트 완료!")
        print("=" * 60)


if __name__ == "__main__":
    # PyTorch 설정
    torch.set_num_threads(mp.cpu_count())  # 모든 CPU 코어 사용
    
    main() 