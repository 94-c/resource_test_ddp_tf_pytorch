"""
PyTorch 메모리 집약적 테스트
이 테스트는 메모리 사용량을 최대화하여 메모리 리소스 모니터링을 테스트합니다.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import sys
import os
import gc
from typing import List, Dict

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.resource_monitor import ResourceMonitor, print_system_info


class MemoryConsumer:
    """메모리 소비자 클래스"""
    def __init__(self):
        self.tensor_storage = []
        self.data_cache = {}
    
    def allocate_large_tensors(self, num_tensors=100, tensor_size=(5000, 5000)):
        """대량의 텐서를 메모리에 할당"""
        print(f"Allocating {num_tensors} large tensors of size {tensor_size}")
        
        tensors = []
        for i in range(num_tensors):
            # 다양한 데이터 타입으로 텐서 생성
            if i % 4 == 0:
                tensor = torch.randn(tensor_size, dtype=torch.float32)
            elif i % 4 == 1:
                tensor = torch.randn(tensor_size, dtype=torch.float64)  # 더 많은 메모리 사용
            elif i % 4 == 2:
                tensor = torch.randint(0, 100, tensor_size, dtype=torch.int64)
            else:
                tensor = torch.ones(tensor_size, dtype=torch.float32)
            
            tensors.append(tensor)
            
            if i % 10 == 0:
                print(f"  Allocated {i+1}/{num_tensors} tensors")
        
        self.tensor_storage.extend(tensors)
        return tensors
    
    def create_memory_intensive_model(self, input_size=10000, hidden_sizes=[8192, 4096, 2048, 1024]):
        """메모리 집약적 모델 생성"""
        print(f"Creating memory intensive model with input size {input_size}")
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1000))  # 출력층
        
        model = nn.Sequential(*layers)
        
        # 모델 파라미터 정보 출력
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model created with {total_params:,} parameters")
        
        return model
    
    def generate_large_dataset(self, num_samples=50000, feature_size=10000):
        """대용량 데이터셋 생성"""
        print(f"Generating large dataset: {num_samples} samples x {feature_size} features")
        
        # 데이터 생성
        X = torch.randn(num_samples, feature_size)
        y = torch.randint(0, 1000, (num_samples,))
        
        # 추가 메모리 사용을 위한 데이터 변환
        X_normalized = torch.nn.functional.normalize(X, dim=1)
        X_squared = X ** 2
        X_expanded = torch.cat([X, X_normalized, X_squared], dim=1)
        
        dataset = {
            'X': X,
            'y': y,
            'X_normalized': X_normalized,
            'X_squared': X_squared,
            'X_expanded': X_expanded
        }
        
        self.data_cache['large_dataset'] = dataset
        return dataset
    
    def progressive_memory_allocation(self, max_iterations=100, size_multiplier=1.1):
        """점진적 메모리 할당"""
        print(f"Progressive memory allocation for {max_iterations} iterations")
        
        current_size = 1000
        allocated_tensors = []
        
        for i in range(max_iterations):
            try:
                # 점진적으로 크기 증가
                tensor_size = int(current_size)
                tensor = torch.randn(tensor_size, tensor_size, dtype=torch.float32)
                allocated_tensors.append(tensor)
                
                current_size *= size_multiplier
                
                if i % 10 == 0:
                    memory_used = sum(t.numel() * t.element_size() for t in allocated_tensors)
                    print(f"  Iteration {i}: Allocated {memory_used / 1024 / 1024:.1f} MB")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  Memory limit reached at iteration {i}")
                    break
                else:
                    raise e
        
        return allocated_tensors
    
    def memory_leak_simulation(self, iterations=50):
        """메모리 누수 시뮬레이션"""
        print(f"Simulating memory leak for {iterations} iterations")
        
        leaked_tensors = []
        
        for i in range(iterations):
            # 매번 새로운 텐서 생성하고 참조 유지
            tensor = torch.randn(2000, 2000, dtype=torch.float32)
            
            # 복잡한 연산 수행
            result = torch.matmul(tensor, tensor.t())
            result = torch.sin(result) + torch.cos(result)
            
            # 결과를 저장 (메모리 누수 시뮬레이션)
            leaked_tensors.append(result)
            
            if i % 10 == 0:
                total_memory = sum(t.numel() * t.element_size() for t in leaked_tensors)
                print(f"  Iteration {i}: Leaked {total_memory / 1024 / 1024:.1f} MB")
        
        return leaked_tensors
    
    def clear_memory(self):
        """메모리 정리"""
        print("Clearing allocated memory...")
        self.tensor_storage.clear()
        self.data_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def batch_processing_memory_test(batch_size=1000, num_batches=100, feature_size=5000):
    """배치 처리 메모리 테스트"""
    print(f"Batch processing memory test: {num_batches} batches of size {batch_size}")
    
    # 메모리 집약적 모델 생성
    model = nn.Sequential(
        nn.Linear(feature_size, 4096),
        nn.ReLU(),
        nn.BatchNorm1d(4096),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 100)
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    batch_results = []
    
    for batch_idx in range(num_batches):
        # 배치 데이터 생성
        X = torch.randn(batch_size, feature_size)
        y = torch.randint(0, 100, (batch_size,))
        
        # 순전파
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        batch_results.append({
            'batch_idx': batch_idx,
            'loss': loss.item(),
            'memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        })
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
    
    return batch_results


def tensor_operations_memory_test(num_operations=1000):
    """텐서 연산 메모리 테스트"""
    print(f"Tensor operations memory test: {num_operations} operations")
    
    # 다양한 크기의 텐서로 연산 수행
    sizes = [1000, 2000, 3000, 4000, 5000]
    operation_results = []
    
    for i in range(num_operations):
        size = sizes[i % len(sizes)]
        
        # 텐서 생성
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        # 메모리 집약적 연산
        c = torch.matmul(a, b)
        d = torch.matmul(c, a.t())
        e = torch.matmul(d, b.t())
        
        # 결과 저장
        result = {
            'operation': i,
            'size': size,
            'result_mean': e.mean().item(),
            'result_std': e.std().item()
        }
        operation_results.append(result)
        
        if i % 100 == 0:
            print(f"  Operation {i}/{num_operations} completed")
    
    return operation_results


def main():
    parser = argparse.ArgumentParser(description='PyTorch 메모리 집약적 테스트')
    parser.add_argument('--duration', type=int, default=120, help='테스트 지속 시간 (초)')
    parser.add_argument('--num-tensors', type=int, default=50, help='할당할 텐서 수')
    parser.add_argument('--tensor-size', type=int, default=3000, help='텐서 크기')
    parser.add_argument('--batch-size', type=int, default=500, help='배치 크기')
    parser.add_argument('--skip-progressive', action='store_true', help='점진적 할당 테스트 건너뛰기')
    parser.add_argument('--skip-leak', action='store_true', help='메모리 누수 테스트 건너뛰기')
    parser.add_argument('--skip-batch', action='store_true', help='배치 처리 테스트 건너뛰기')
    
    args = parser.parse_args()
    
    # 시스템 정보 출력
    print_system_info()
    
    # 리소스 모니터링 시작
    monitor = ResourceMonitor(interval=0.5)
    monitor.start_monitoring()
    
    print(f"\n🧠 PyTorch 메모리 집약적 테스트 시작 (지속 시간: {args.duration}초)")
    print("=" * 60)
    
    # 메모리 소비자 객체 생성
    memory_consumer = MemoryConsumer()
    
    start_time = time.time()
    
    try:
        # 테스트 실행
        test_results = {}
        
        # 1. 대량 텐서 할당
        print("\n1. 대량 텐서 할당 테스트")
        tensors = memory_consumer.allocate_large_tensors(
            num_tensors=args.num_tensors,
            tensor_size=(args.tensor_size, args.tensor_size)
        )
        test_results['large_tensors'] = len(tensors)
        
        # 2. 메모리 집약적 모델 생성
        print("\n2. 메모리 집약적 모델 생성")
        model = memory_consumer.create_memory_intensive_model()
        test_results['model_params'] = sum(p.numel() for p in model.parameters())
        
        # 3. 대용량 데이터셋 생성
        print("\n3. 대용량 데이터셋 생성")
        dataset = memory_consumer.generate_large_dataset(
            num_samples=20000,
            feature_size=5000
        )
        test_results['dataset_size'] = dataset['X'].shape
        
        # 4. 점진적 메모리 할당
        if not args.skip_progressive:
            print("\n4. 점진적 메모리 할당")
            progressive_tensors = memory_consumer.progressive_memory_allocation(max_iterations=50)
            test_results['progressive_tensors'] = len(progressive_tensors)
        
        # 5. 메모리 누수 시뮬레이션
        if not args.skip_leak:
            print("\n5. 메모리 누수 시뮬레이션")
            leaked_tensors = memory_consumer.memory_leak_simulation(iterations=30)
            test_results['leaked_tensors'] = len(leaked_tensors)
        
        # 6. 배치 처리 테스트
        if not args.skip_batch:
            print("\n6. 배치 처리 테스트")
            batch_results = batch_processing_memory_test(
                batch_size=args.batch_size,
                num_batches=50,
                feature_size=2000
            )
            test_results['batch_results'] = len(batch_results)
        
        # 7. 텐서 연산 테스트
        print("\n7. 텐서 연산 테스트")
        operation_results = tensor_operations_memory_test(num_operations=200)
        test_results['operation_results'] = len(operation_results)
        
        # 남은 시간 동안 추가 텐서 할당
        elapsed_time = time.time() - start_time
        remaining_time = args.duration - elapsed_time
        
        if remaining_time > 10:
            print(f"\n8. 추가 텐서 할당 ({remaining_time:.1f}초 동안)")
            extra_tensors = memory_consumer.allocate_large_tensors(
                num_tensors=max(10, int(remaining_time / 5)),
                tensor_size=(2000, 2000)
            )
            test_results['extra_tensors'] = len(extra_tensors)
        
    except KeyboardInterrupt:
        print("\n테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 리소스 모니터링 종료
        monitor.stop_monitoring()
        monitor.print_summary()
        
        # 최종 리소스 사용량 확인
        final_usage = monitor.get_current_usage()
        print(f"\n최종 리소스 사용량:")
        print(f"Memory: {final_usage.get('memory_percent', 0):.1f}%")
        print(f"Memory MB: {final_usage.get('memory_mb', 0):.1f} MB")
        
        # 메모리 정리
        print("\n메모리 정리 중...")
        memory_consumer.clear_memory()
        
        print("\n🎯 PyTorch 메모리 집약적 테스트 완료!")
        print("=" * 60)


if __name__ == "__main__":
    main() 