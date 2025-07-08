"""
PyTorch GPU 메모리 집약적 테스트
이 테스트는 GPU 메모리 사용량을 최대화하여 GPU 메모리 리소스 모니터링을 테스트합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse
import sys
import os
import gc
from typing import List, Dict, Optional

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.resource_monitor import ResourceMonitor, print_system_info


class GPUMemoryConsumer:
    """GPU 메모리 소비자 클래스"""
    def __init__(self):
        self.gpu_tensors = []
        self.gpu_models = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA가 사용 불가능합니다. CPU 모드로 실행됩니다.")
        else:
            print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name()}")
            print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def allocate_large_gpu_tensors(self, num_tensors=50, tensor_size=(4000, 4000)):
        """대량의 텐서를 GPU 메모리에 할당"""
        if not torch.cuda.is_available():
            print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
            return []
        
        print(f"GPU에 {num_tensors}개의 대용량 텐서 할당 (크기: {tensor_size})")
        
        tensors = []
        for i in range(num_tensors):
            try:
                # 다양한 데이터 타입으로 GPU 텐서 생성
                if i % 3 == 0:
                    tensor = torch.randn(tensor_size, dtype=torch.float32, device=self.device)
                elif i % 3 == 1:
                    tensor = torch.randn(tensor_size, dtype=torch.float16, device=self.device)  # 메모리 절약
                else:
                    tensor = torch.randint(0, 100, tensor_size, dtype=torch.int32, device=self.device)
                
                tensors.append(tensor)
                
                if i % 10 == 0:
                    allocated_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"  할당됨 {i+1}/{num_tensors}, GPU 메모리: {allocated_memory:.2f} GB")
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  GPU 메모리 부족으로 {i}번째 텐서에서 중단")
                    break
                else:
                    raise e
        
        self.gpu_tensors.extend(tensors)
        return tensors
    
    def create_massive_gpu_model(self, input_size=20000):
        """대용량 GPU 모델 생성"""
        if not torch.cuda.is_available():
            print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
            return None
        
        print(f"대용량 GPU 모델 생성 (입력 크기: {input_size})")
        
        class MassiveGPUModel(nn.Module):
            def __init__(self, input_size):
                super(MassiveGPUModel, self).__init__()
                
                # 매우 큰 레이어들
                self.layers = nn.ModuleList([
                    nn.Linear(input_size, 16384),
                    nn.Linear(16384, 8192),
                    nn.Linear(8192, 4096),
                    nn.Linear(4096, 2048),
                    nn.Linear(2048, 1024),
                    nn.Linear(1024, 512),
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 64),
                    nn.Linear(64, 10)
                ])
                
                # 배치 정규화 레이어들
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(16384),
                    nn.BatchNorm1d(8192),
                    nn.BatchNorm1d(4096),
                    nn.BatchNorm1d(2048),
                    nn.BatchNorm1d(1024),
                    nn.BatchNorm1d(512),
                    nn.BatchNorm1d(256),
                    nn.BatchNorm1d(128),
                    nn.BatchNorm1d(64)
                ])
                
                # 드롭아웃 레이어들
                self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(9)])
            
            def forward(self, x):
                for i, (layer, bn, dropout) in enumerate(zip(self.layers[:-1], self.batch_norms, self.dropouts)):
                    x = layer(x)
                    x = F.relu(x)
                    x = bn(x)
                    x = dropout(x)
                
                x = self.layers[-1](x)  # 마지막 레이어
                return x
        
        try:
            model = MassiveGPUModel(input_size).to(self.device)
            total_params = sum(p.numel() for p in model.parameters())
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
            
            print(f"  모델 생성 완료: {total_params:,} 파라미터, {model_size_mb:.1f} MB")
            
            self.gpu_models.append(model)
            return model
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  GPU 메모리 부족으로 모델 생성 실패")
                return None
            else:
                raise e
    
    def progressive_gpu_memory_allocation(self, max_iterations=100, size_multiplier=1.05):
        """점진적 GPU 메모리 할당"""
        if not torch.cuda.is_available():
            print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
            return []
        
        print(f"점진적 GPU 메모리 할당 ({max_iterations} 반복)")
        
        current_size = 1000
        allocated_tensors = []
        
        for i in range(max_iterations):
            try:
                tensor_size = int(current_size)
                tensor = torch.randn(tensor_size, tensor_size, dtype=torch.float32, device=self.device)
                allocated_tensors.append(tensor)
                
                current_size *= size_multiplier
                
                if i % 10 == 0:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"  반복 {i}: GPU 메모리 사용량 {gpu_memory:.2f} GB")
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  GPU 메모리 한계에 도달: 반복 {i}")
                    break
                else:
                    raise e
        
        return allocated_tensors
    
    def gpu_memory_stress_test(self, stress_level=5):
        """GPU 메모리 스트레스 테스트"""
        if not torch.cuda.is_available():
            print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
            return
        
        print(f"GPU 메모리 스트레스 테스트 (레벨: {stress_level})")
        
        # 스트레스 레벨에 따른 텐서 크기 조정
        base_size = 2000
        tensor_size = base_size * stress_level
        
        stress_tensors = []
        
        for i in range(20):
            try:
                # 다양한 연산으로 GPU 메모리 사용
                a = torch.randn(tensor_size, tensor_size, device=self.device)
                b = torch.randn(tensor_size, tensor_size, device=self.device)
                
                # 행렬 곱셈
                c = torch.matmul(a, b)
                
                # 추가 연산
                d = torch.sin(c) + torch.cos(c)
                e = torch.matmul(d, a.t())
                
                stress_tensors.append(e)
                
                if i % 5 == 0:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"  스트레스 {i}: GPU 메모리 {gpu_memory:.2f} GB")
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  스트레스 테스트 중 메모리 부족: 단계 {i}")
                    break
                else:
                    raise e
        
        return stress_tensors
    
    def multi_gpu_memory_test(self):
        """다중 GPU 메모리 테스트"""
        if not torch.cuda.is_available():
            print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
            return
        
        gpu_count = torch.cuda.device_count()
        print(f"다중 GPU 메모리 테스트 ({gpu_count}개 GPU)")
        
        if gpu_count < 2:
            print("단일 GPU 환경에서 다중 GPU 시뮬레이션")
            gpu_count = 1
        
        gpu_tensors = {}
        
        for gpu_id in range(gpu_count):
            try:
                device = torch.device(f'cuda:{gpu_id}')
                print(f"  GPU {gpu_id}에 텐서 할당 중...")
                
                tensors = []
                for i in range(10):
                    tensor = torch.randn(3000, 3000, device=device)
                    tensors.append(tensor)
                
                gpu_tensors[gpu_id] = tensors
                
                gpu_memory = torch.cuda.memory_allocated(gpu_id) / 1024**3
                print(f"  GPU {gpu_id}: {gpu_memory:.2f} GB 할당됨")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  GPU {gpu_id} 메모리 부족")
                else:
                    raise e
        
        return gpu_tensors
    
    def clear_gpu_memory(self):
        """GPU 메모리 정리"""
        print("GPU 메모리 정리 중...")
        
        self.gpu_tensors.clear()
        self.gpu_models.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            print(f"정리 후 GPU 메모리: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


def gpu_training_memory_test(batch_size=256, num_epochs=20, model_size='large'):
    """GPU 학습 메모리 테스트"""
    if not torch.cuda.is_available():
        print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
        return
    
    device = torch.device('cuda')
    print(f"GPU 학습 메모리 테스트 (배치: {batch_size}, 에포크: {num_epochs})")
    
    # 모델 크기에 따른 설정
    if model_size == 'large':
        input_size, hidden_size = 5000, 2048
    elif model_size == 'medium':
        input_size, hidden_size = 2000, 1024
    else:
        input_size, hidden_size = 1000, 512
    
    # 모델 생성
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_size),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_size // 2),
        nn.Linear(hidden_size // 2, hidden_size // 4),
        nn.ReLU(),
        nn.Linear(hidden_size // 4, 100)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 학습 루프
    for epoch in range(num_epochs):
        epoch_memory_usage = []
        
        for batch_idx in range(20):  # 에포크당 20 배치
            # 배치 데이터 생성
            X = torch.randn(batch_size, input_size, device=device)
            y = torch.randint(0, 100, (batch_size,), device=device)
            
            # 순전파
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            # GPU 메모리 사용량 기록
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            epoch_memory_usage.append(gpu_memory)
        
        avg_memory = np.mean(epoch_memory_usage)
        max_memory = np.max(epoch_memory_usage)
        
        if epoch % 5 == 0:
            print(f"  에포크 {epoch}: 평균 GPU 메모리 {avg_memory:.2f} GB, 최대 {max_memory:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description='PyTorch GPU 메모리 집약적 테스트')
    parser.add_argument('--duration', type=int, default=180, help='테스트 지속 시간 (초)')
    parser.add_argument('--num-tensors', type=int, default=30, help='할당할 GPU 텐서 수')
    parser.add_argument('--tensor-size', type=int, default=3000, help='텐서 크기')
    parser.add_argument('--stress-level', type=int, default=3, help='스트레스 테스트 레벨 (1-5)')
    parser.add_argument('--skip-progressive', action='store_true', help='점진적 할당 테스트 건너뛰기')
    parser.add_argument('--skip-stress', action='store_true', help='스트레스 테스트 건너뛰기')
    parser.add_argument('--skip-training', action='store_true', help='학습 테스트 건너뛰기')
    parser.add_argument('--skip-multi-gpu', action='store_true', help='다중 GPU 테스트 건너뛰기')
    
    args = parser.parse_args()
    
    # 시스템 정보 출력
    print_system_info()
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"\n🎮 GPU 정보:")
        print(f"  디바이스: {torch.cuda.get_device_name()}")
        print(f"  메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  GPU 개수: {torch.cuda.device_count()}")
    
    # 리소스 모니터링 시작
    monitor = ResourceMonitor(interval=0.5)
    monitor.start_monitoring()
    
    print(f"\n🎯 PyTorch GPU 메모리 집약적 테스트 시작 (지속 시간: {args.duration}초)")
    print("=" * 60)
    
    # GPU 메모리 소비자 생성
    gpu_consumer = GPUMemoryConsumer()
    
    start_time = time.time()
    
    try:
        # 테스트 실행
        test_results = {}
        
        # 1. 대량 GPU 텐서 할당
        print("\n1. 대량 GPU 텐서 할당 테스트")
        gpu_tensors = gpu_consumer.allocate_large_gpu_tensors(
            num_tensors=args.num_tensors,
            tensor_size=(args.tensor_size, args.tensor_size)
        )
        test_results['gpu_tensors'] = len(gpu_tensors)
        
        # 2. 대용량 GPU 모델 생성
        print("\n2. 대용량 GPU 모델 생성")
        gpu_model = gpu_consumer.create_massive_gpu_model(input_size=10000)
        test_results['gpu_model'] = gpu_model is not None
        
        # 3. 점진적 GPU 메모리 할당
        if not args.skip_progressive:
            print("\n3. 점진적 GPU 메모리 할당")
            progressive_tensors = gpu_consumer.progressive_gpu_memory_allocation(max_iterations=50)
            test_results['progressive_tensors'] = len(progressive_tensors)
        
        # 4. GPU 메모리 스트레스 테스트
        if not args.skip_stress:
            print("\n4. GPU 메모리 스트레스 테스트")
            stress_tensors = gpu_consumer.gpu_memory_stress_test(stress_level=args.stress_level)
            test_results['stress_tensors'] = len(stress_tensors) if stress_tensors else 0
        
        # 5. 다중 GPU 메모리 테스트
        if not args.skip_multi_gpu:
            print("\n5. 다중 GPU 메모리 테스트")
            multi_gpu_tensors = gpu_consumer.multi_gpu_memory_test()
            test_results['multi_gpu_tensors'] = len(multi_gpu_tensors) if multi_gpu_tensors else 0
        
        # 6. GPU 학습 메모리 테스트
        if not args.skip_training:
            print("\n6. GPU 학습 메모리 테스트")
            gpu_training_memory_test(batch_size=128, num_epochs=15, model_size='large')
            test_results['training_completed'] = True
        
        # 남은 시간 동안 추가 GPU 텐서 할당
        elapsed_time = time.time() - start_time
        remaining_time = args.duration - elapsed_time
        
        if remaining_time > 15:
            print(f"\n7. 추가 GPU 텐서 할당 ({remaining_time:.1f}초 동안)")
            extra_tensors = gpu_consumer.allocate_large_gpu_tensors(
                num_tensors=max(5, int(remaining_time / 10)),
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
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.memory_allocated(i) / 1024**3
                print(f"GPU {i} 메모리: {gpu_memory:.2f} GB")
        
        # GPU 메모리 정리
        print("\nGPU 메모리 정리 중...")
        gpu_consumer.clear_gpu_memory()
        
        print("\n🎯 PyTorch GPU 메모리 집약적 테스트 완료!")
        print("=" * 60)


if __name__ == "__main__":
    main() 