"""
PyTorch GPU 사용률 집약적 테스트 (DCGM 메트릭 대응)
이 테스트는 GPU 사용률을 최대화하여 DCGM_FI_PROF_GR_ENGINE_ACTIVE 메트릭에 나타나도록 합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import argparse
import sys
import os
import threading
import concurrent.futures
from typing import List, Dict, Optional
import signal
import multiprocessing as mp

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.resource_monitor import ResourceMonitor, print_system_info


class IntensiveGPUWorkload:
    """DCGM 메트릭에 나타나도록 하는 집약적 GPU 워크로드"""
    
    def __init__(self, duration_minutes=10):
        self.duration_minutes = duration_minutes
        self.duration_seconds = duration_minutes * 60
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.stop_event = threading.Event()
        self.workload_threads = []
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA가 사용 불가능합니다. CPU 모드로 실행됩니다.")
            return
        
        print(f"✅ CUDA 사용 가능: {self.device_count}개 GPU 감지")
        for i in range(self.device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    def create_intensive_model(self, device_id):
        """GPU 집약적 모델 생성"""
        class IntensiveModel(nn.Module):
            def __init__(self):
                super(IntensiveModel, self).__init__()
                # 매우 큰 모델로 GPU 사용률 최대화
                self.conv_layers = nn.ModuleList([
                    nn.Conv2d(3, 128, kernel_size=7, padding=3),
                    nn.Conv2d(128, 256, kernel_size=5, padding=2),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                    nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
                    nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
                    nn.Conv2d(1024, 512, kernel_size=3, padding=1),
                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                    nn.Conv2d(256, 128, kernel_size=3, padding=1),
                    nn.Conv2d(128, 64, kernel_size=3, padding=1),
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.Conv2d(32, 3, kernel_size=3, padding=1)
                ])
                
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm2d(128),
                    nn.BatchNorm2d(256),
                    nn.BatchNorm2d(512),
                    nn.BatchNorm2d(1024),
                    nn.BatchNorm2d(2048),
                    nn.BatchNorm2d(1024),
                    nn.BatchNorm2d(512),
                    nn.BatchNorm2d(256),
                    nn.BatchNorm2d(128),
                    nn.BatchNorm2d(64),
                    nn.BatchNorm2d(32),
                ])
                
                # 트랜스포머 레이어 추가
                self.transformer_layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, batch_first=True)
                    for _ in range(6)
                ])
                
                # Dense 레이어들
                self.dense_layers = nn.ModuleList([
                    nn.Linear(512, 2048),
                    nn.Linear(2048, 4096),
                    nn.Linear(4096, 2048),
                    nn.Linear(2048, 512),
                    nn.Linear(512, 100)
                ])
                
            def forward(self, x):
                # 컨볼루션 레이어들
                for i, (conv, bn) in enumerate(zip(self.conv_layers[:-1], self.batch_norms)):
                    x = conv(x)
                    x = F.relu(x)
                    x = bn(x)
                    
                    # 다운샘플링과 업샘플링으로 계산량 증가
                    if i < 5:
                        x = F.max_pool2d(x, 2)
                    elif i >= 6:
                        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                
                x = self.conv_layers[-1](x)
                
                # 글로벌 평균 풀링
                x = F.adaptive_avg_pool2d(x, (16, 16))
                
                # 트랜스포머 처리를 위한 reshape
                batch_size = x.shape[0]
                x = x.view(batch_size, -1, 512)  # (batch, seq_len, d_model)
                
                # 트랜스포머 레이어들
                for transformer in self.transformer_layers:
                    x = transformer(x)
                
                # 평균 풀링
                x = x.mean(dim=1)  # (batch, d_model)
                
                # Dense 레이어들
                for dense in self.dense_layers:
                    x = F.relu(dense(x))
                
                return x
        
        device = torch.device(f'cuda:{device_id}')
        model = IntensiveModel().to(device)
        return model, device
    
    def continuous_gpu_workload(self, device_id, workload_id):
        """지속적인 GPU 워크로드 실행"""
        print(f"🚀 GPU {device_id} 워크로드 {workload_id} 시작")
        
        try:
            torch.cuda.set_device(device_id)
            model, device = self.create_intensive_model(device_id)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # 큰 배치 크기로 GPU 사용률 최대화
            batch_size = 32
            start_time = time.time()
            iteration = 0
            
            while not self.stop_event.is_set() and (time.time() - start_time) < self.duration_seconds:
                try:
                    # 큰 이미지 데이터 생성
                    input_data = torch.randn(batch_size, 3, 512, 512, device=device)
                    target = torch.randint(0, 100, (batch_size,), device=device)
                    
                    # 순전파
                    optimizer.zero_grad()
                    output = model(input_data)
                    loss = criterion(output, target)
                    
                    # 역전파
                    loss.backward()
                    optimizer.step()
                    
                    # 추가 계산 집약적 연산들
                    if iteration % 5 == 0:
                        # 대용량 행렬 연산
                        matrix_a = torch.randn(2048, 2048, device=device)
                        matrix_b = torch.randn(2048, 2048, device=device)
                        result = torch.matmul(matrix_a, matrix_b)
                        
                        # 고유값 분해 (매우 집약적)
                        try:
                            eigenvalues = torch.linalg.eigvals(result[:1024, :1024])
                        except:
                            pass
                        
                        # FFT 연산
                        fft_input = torch.randn(8192, device=device)
                        fft_result = torch.fft.fft(fft_input)
                        ifft_result = torch.fft.ifft(fft_result)
                    
                    # 메모리 압박 방지를 위한 주기적 정리
                    if iteration % 20 == 0:
                        torch.cuda.empty_cache()
                        elapsed = time.time() - start_time
                        remaining = self.duration_seconds - elapsed
                        print(f"  GPU {device_id} 워크로드 {workload_id}: {iteration} 반복 완료, "
                              f"남은 시간: {remaining:.1f}초, Loss: {loss.item():.4f}")
                    
                    iteration += 1
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"  GPU {device_id} 메모리 부족, 배치 크기 줄임")
                    batch_size = max(1, batch_size // 2)
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    print(f"  GPU {device_id} 워크로드 오류: {e}")
                    continue
            
            print(f"✅ GPU {device_id} 워크로드 {workload_id} 완료 ({iteration} 반복)")
            
        except Exception as e:
            print(f"❌ GPU {device_id} 워크로드 {workload_id} 실패: {e}")
    
    def start_workloads(self):
        """모든 GPU에서 워크로드 시작"""
        if not torch.cuda.is_available():
            print("GPU가 사용 불가능합니다.")
            return
        
        print(f"🎯 {self.device_count}개 GPU에서 {self.duration_minutes}분 동안 집약적 워크로드 시작")
        print("   DCGM_FI_PROF_GR_ENGINE_ACTIVE 메트릭에 나타날 때까지 기다려주세요...")
        
        # 각 GPU에서 여러 워크로드 실행
        for device_id in range(self.device_count):
            # GPU당 2개의 워크로드 스레드 실행
            for workload_id in range(2):
                thread = threading.Thread(
                    target=self.continuous_gpu_workload,
                    args=(device_id, workload_id)
                )
                thread.daemon = True
                thread.start()
                self.workload_threads.append(thread)
        
        # 시그널 핸들러 등록
        def signal_handler(signum, frame):
            print(f"\n🛑 중단 신호 받음. 워크로드 중지 중...")
            self.stop_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 워크로드 완료 대기
        start_time = time.time()
        try:
            while (time.time() - start_time) < self.duration_seconds and not self.stop_event.is_set():
                time.sleep(10)
                elapsed = time.time() - start_time
                remaining = self.duration_seconds - elapsed
                print(f"⏱️  진행 중... 남은 시간: {remaining:.1f}초")
        except KeyboardInterrupt:
            print(f"\n🛑 사용자 중단 요청")
            self.stop_event.set()
        
        # 모든 스레드 종료 대기
        self.stop_event.set()
        for thread in self.workload_threads:
            thread.join(timeout=5)
        
        print(f"�� 모든 GPU 워크로드 완료")


def main():
    parser = argparse.ArgumentParser(description='PyTorch GPU 사용률 집약적 테스트 (DCGM 메트릭 대응)')
    parser.add_argument('--duration', type=int, default=600, help='테스트 지속 시간 (초)')
    
    args = parser.parse_args()
    
    # 시스템 정보 출력
    print_system_info()
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"\n🎮 GPU 정보:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            print(f"    멀티프로세서: {torch.cuda.get_device_properties(i).multi_processor_count}")
    
    # 리소스 모니터링 시작
    monitor = ResourceMonitor(interval=1)
    monitor.start_monitoring()
    
    duration_minutes = max(1, args.duration // 60)
    print(f"\n🚀 PyTorch GPU 사용률 집약적 테스트 시작 ({duration_minutes}분)")
    print("   DCGM_FI_PROF_GR_ENGINE_ACTIVE 메트릭에 나타날 때까지 기다려주세요...")
    print("=" * 60)
    
    # GPU 사용률 최대화 객체 생성
    gpu_workload = IntensiveGPUWorkload(duration_minutes=duration_minutes)
    
    start_time = time.time()
    
    try:
        # 집약적 GPU 워크로드 시작
        gpu_workload.start_workloads()
        
    except KeyboardInterrupt:
        print("\n🛑 테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 리소스 모니터링 종료
        monitor.stop_monitoring()
        monitor.print_summary()
        
        # 최종 리소스 사용량 확인
        final_usage = monitor.get_current_usage()
        print(f"\n📊 최종 리소스 사용량:")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_utilization = final_usage.get(f'gpu_{i}_utilization', 0)
                gpu_memory = final_usage.get(f'gpu_{i}_memory_used', 0)
                print(f"  GPU {i} 사용률: {gpu_utilization}%")
                print(f"  GPU {i} 메모리: {gpu_memory:.2f} GB")
        
        # GPU 메모리 정리
        print("\n🧹 GPU 메모리 정리 중...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        total_duration = time.time() - start_time
        print(f"\n🎯 PyTorch GPU 사용률 집약적 테스트 완료!")
        print(f"   총 실행 시간: {total_duration:.1f}초")
        print("=" * 60)


if __name__ == "__main__":
    main() 