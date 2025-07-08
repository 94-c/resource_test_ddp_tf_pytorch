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
            # GPU 메모리 정보 출력 (디버깅용)
            try:
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"    총 메모리: {gpu_memory:.1f} GB")
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    cached = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"    할당된 메모리: {allocated:.2f} GB, 캐시된 메모리: {cached:.2f} GB")
            except Exception as e:
                print(f"    메모리 정보 조회 실패: {e}")
    
    def create_intensive_model(self, device_id):
        """GPU 집약적 모델 생성 (컨테이너 환경 최적화)"""
        class SafeIntensiveModel(nn.Module):
            def __init__(self):
                super(SafeIntensiveModel, self).__init__()
                # 메모리 사용량을 줄인 안전한 모델
                self.conv_layers = nn.ModuleList([
                    nn.Conv2d(3, 64, kernel_size=5, padding=2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                    nn.Conv2d(256, 128, kernel_size=3, padding=1),
                    nn.Conv2d(128, 64, kernel_size=3, padding=1),
                    nn.Conv2d(64, 32, kernel_size=3, padding=1),
                    nn.Conv2d(32, 3, kernel_size=3, padding=1)
                ])
                
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm2d(64),
                    nn.BatchNorm2d(128),
                    nn.BatchNorm2d(256),
                    nn.BatchNorm2d(512),
                    nn.BatchNorm2d(256),
                    nn.BatchNorm2d(128),
                    nn.BatchNorm2d(64),
                    nn.BatchNorm2d(32),
                ])
                
                # 더 작은 Dense 레이어들
                self.dense_layers = nn.ModuleList([
                    nn.Linear(256, 512),
                    nn.Linear(512, 256),
                    nn.Linear(256, 100)
                ])
                
            def forward(self, x):
                # 컨볼루션 레이어들
                for i, (conv, bn) in enumerate(zip(self.conv_layers[:-1], self.batch_norms)):
                    x = conv(x)
                    x = F.relu(x)
                    x = bn(x)
                    
                    # 적절한 다운샘플링
                    if i in [1, 3]:
                        x = F.max_pool2d(x, 2)
                    elif i in [5, 7]:
                        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                
                x = self.conv_layers[-1](x)
                
                # 글로벌 평균 풀링
                x = F.adaptive_avg_pool2d(x, (4, 4))
                x = x.view(x.size(0), -1)
                
                # Dense 레이어들
                for dense in self.dense_layers:
                    x = F.relu(dense(x))
                
                return x
        
        device = torch.device(f'cuda:{device_id}')
        model = SafeIntensiveModel().to(device)
        return model, device
    
    def continuous_gpu_workload(self, device_id, workload_id):
        """지속적인 GPU 워크로드 실행 (안전 버전)"""
        print(f"🚀 GPU {device_id} 워크로드 {workload_id} 시작")
        
        try:
            torch.cuda.set_device(device_id)
            
            # 메모리 정리
            torch.cuda.empty_cache()
            
            model, device = self.create_intensive_model(device_id)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # 더 작은 배치 크기로 시작
            batch_size = 8
            start_time = time.time()
            iteration = 0
            
            while not self.stop_event.is_set() and (time.time() - start_time) < self.duration_seconds:
                try:
                    # 더 작은 이미지 데이터 생성
                    input_data = torch.randn(batch_size, 3, 256, 256, device=device)
                    target = torch.randint(0, 100, (batch_size,), device=device)
                    
                    # 순전파
                    optimizer.zero_grad()
                    output = model(input_data)
                    loss = criterion(output, target)
                    
                    # 역전파
                    loss.backward()
                    optimizer.step()
                    
                    # 더 집약적이지만 안전한 연산들
                    if iteration % 3 == 0:
                        # 중간 크기 행렬 연산
                        matrix_a = torch.randn(1024, 1024, device=device)
                        matrix_b = torch.randn(1024, 1024, device=device)
                        result = torch.matmul(matrix_a, matrix_b)
                        
                        # 간단한 FFT 연산
                        fft_input = torch.randn(4096, device=device)
                        fft_result = torch.fft.fft(fft_input)
                        ifft_result = torch.fft.ifft(fft_result)
                    
                    # 더 빈번한 메모리 정리 (5회마다)
                    if iteration % 5 == 0:
                        torch.cuda.empty_cache()
                        elapsed = time.time() - start_time
                        remaining = self.duration_seconds - elapsed
                        
                        # 메모리 사용량 확인
                        try:
                            allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                            reserved = torch.cuda.memory_reserved(device_id) / 1024**3
                            print(f"  GPU {device_id} 워크로드 {workload_id}: {iteration} 반복 완료, "
                                  f"남은 시간: {remaining:.1f}초, Loss: {loss.item():.4f}, "
                                  f"메모리: {allocated:.2f}GB/{reserved:.2f}GB")
                        except Exception:
                            print(f"  GPU {device_id} 워크로드 {workload_id}: {iteration} 반복 완료, "
                                  f"남은 시간: {remaining:.1f}초, Loss: {loss.item():.4f}")
                    
                    iteration += 1
                    
                except torch.cuda.OutOfMemoryError as e:
                    print(f"  GPU {device_id} 메모리 부족, 배치 크기 줄임: {batch_size} -> {max(1, batch_size // 2)}")
                    batch_size = max(1, batch_size // 2)
                    torch.cuda.empty_cache()
                    time.sleep(1)  # 잠시 대기
                    continue
                except RuntimeError as e:
                    if "NVML" in str(e) or "CUDA" in str(e):
                        print(f"  GPU {device_id} CUDA/NVML 오류 발생: {e}")
                        print(f"  워크로드를 안전하게 종료합니다.")
                        break
                    else:
                        print(f"  GPU {device_id} 런타임 오류: {e}")
                        continue
                except Exception as e:
                    print(f"  GPU {device_id} 워크로드 오류: {e}")
                    continue
            
            print(f"✅ GPU {device_id} 워크로드 {workload_id} 완료 ({iteration} 반복)")
            
        except Exception as e:
            print(f"❌ GPU {device_id} 워크로드 {workload_id} 실패: {e}")
        finally:
            # 최종 메모리 정리
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    
    def start_workloads(self):
        """모든 GPU에서 워크로드 시작 (안전 버전)"""
        if not torch.cuda.is_available() or self.device_count == 0:
            print("GPU가 사용 불가능합니다.")
            return
        
        print(f"🎯 {self.device_count}개 GPU에서 {self.duration_minutes}분 동안 집약적 워크로드 시작")
        print("   DCGM_FI_PROF_GR_ENGINE_ACTIVE 메트릭에 나타날 때까지 기다려주세요...")
        print("   ⚠️  컨테이너 환경에서 안전 모드로 실행됩니다.")
        
        # 각 GPU에서 단일 워크로드만 실행 (안전성 향상)
        for device_id in range(self.device_count):
            thread = threading.Thread(
                target=self.continuous_gpu_workload,
                args=(device_id, 0)
            )
            thread.daemon = True
            thread.start()
            self.workload_threads.append(thread)
            
            # GPU 간 시작 간격 (리소스 경합 방지)
            time.sleep(2)
        
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
                
                # 주기적으로 전체 GPU 메모리 상태 확인
                try:
                    for i in range(self.device_count):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        print(f"    GPU {i} 메모리: {allocated:.2f}GB 할당, {reserved:.2f}GB 예약")
                except Exception:
                    pass
                    
        except KeyboardInterrupt:
            print(f"\n🛑 사용자 중단 요청")
            self.stop_event.set()
        
        # 모든 스레드 종료 대기
        self.stop_event.set()
        for thread in self.workload_threads:
            thread.join(timeout=10)
        
        # 최종 메모리 정리
        try:
            for i in range(self.device_count):
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        print(f"🏁 모든 GPU 워크로드 완료")


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