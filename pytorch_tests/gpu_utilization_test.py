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
        self.stop_event = threading.Event()
        self.workload_threads = []
        
        # 안전한 GPU 감지
        self.device_count = 0
        self.available_devices = []
        self.cuda_available = False
        
        # CUDA 기본 사용 가능성 확인
        try:
            if not torch.cuda.is_available():
                print("⚠️  CUDA가 사용 불가능합니다. CPU 모드로 실행됩니다.")
                return
                
            # PyTorch CUDA 초기화 안전성 확인
            try:
                torch.cuda.init()
                potential_devices = torch.cuda.device_count()
                print(f"🔍 감지된 GPU 개수: {potential_devices}")
            except Exception as e:
                print(f"❌ CUDA 초기화 실패: {e}")
                print("⚠️  CPU 모드로 전환됩니다.")
                return
                
        except Exception as e:
            print(f"❌ CUDA 확인 중 오류: {e}")
            print("⚠️  CPU 모드로 실행됩니다.")
            return
        
        # 안전한 GPU 장치 감지
        for i in range(potential_devices):
            try:
                print(f"🔍 GPU {i} 접근성 테스트 중...")
                
                # 단계별 안전한 GPU 접근 테스트
                # 1단계: 장치 속성 확인
                try:
                    props = torch.cuda.get_device_properties(i)
                    device_name = props.name
                    total_memory = props.total_memory
                except Exception as e:
                    print(f"❌ GPU {i}: 장치 속성 조회 실패 - {e}")
                    continue
                
                # 2단계: 컨텍스트 생성 및 메모리 할당 테스트
                try:
                    with torch.cuda.device(i):
                        # 매우 작은 메모리 할당으로 테스트
                        test_tensor = torch.zeros(1, device=f'cuda:{i}', dtype=torch.float32)
                        
                        # 간단한 연산 테스트
                        result = test_tensor + 1.0
                        
                        # 동기화 테스트 (이 부분에서 많은 오류 발생)
                        torch.cuda.synchronize(device=i)
                        
                        # 메모리 정리
                        del test_tensor, result
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"❌ GPU {i}: 메모리/연산 테스트 실패 - {str(e)[:100]}...")
                    if "device >= 0 && device < num_gpus" in str(e):
                        print(f"  → MIG 환경 또는 GPU 매핑 문제")
                    elif "CUDA out of memory" in str(e):
                        print(f"  → GPU 메모리 부족")
                    elif "no kernel image" in str(e):
                        print(f"  → CUDA 버전 호환성 문제")
                    continue
                
                # 모든 테스트 통과 시 사용 가능한 장치로 추가
                self.available_devices.append(i)
                print(f"✅ GPU {i}: 접근 가능")
                
                # GPU 정보 출력 (안전한 방식)
                try:
                    print(f"  이름: {device_name}")
                    print(f"  총 메모리: {total_memory / 1024**3:.1f} GB")
                    print(f"  멀티프로세서: {props.multi_processor_count}")
                    
                    # 현재 메모리 사용량 (오류 발생 시 건너뛰기)
                    try:
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        cached = torch.cuda.memory_reserved(i) / 1024**3
                        print(f"  할당된 메모리: {allocated:.2f} GB, 캐시된 메모리: {cached:.2f} GB")
                    except:
                        pass
                        
                except Exception as e:
                    print(f"  정보 출력 중 오류 (하지만 GPU는 사용 가능): {e}")
                    
            except Exception as e:
                print(f"❌ GPU {i}: 예상치 못한 오류 - {str(e)[:100]}...")
                continue
        
        self.device_count = len(self.available_devices)
        
        if self.device_count == 0:
            print("❌ 사용 가능한 GPU가 없습니다.")
            self.cuda_available = False
        else:
            print(f"✅ 사용 가능한 GPU: {self.device_count}개 (인덱스: {self.available_devices})")
            self.cuda_available = True
    
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
                
                # 마지막 컨볼루션 출력 계산: 32 channels × 4 × 4 = 512
                self.final_conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
                
                # 차원 맞춘 Dense 레이어들 (32 × 4 × 4 = 512)
                self.dense_layers = nn.ModuleList([
                    nn.Linear(512, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 100)
                ])
                
            def forward(self, x):
                # 컨볼루션 레이어들
                for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
                    x = conv(x)
                    x = F.relu(x)
                    x = bn(x)
                    
                    # 적절한 다운샘플링
                    if i in [1, 3]:
                        x = F.max_pool2d(x, 2)
                    elif i in [5, 7]:
                        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                
                # 마지막 컨볼루션
                x = self.final_conv(x)
                x = F.relu(x)
                
                # 글로벌 평균 풀링 (32 × 4 × 4 = 512)
                x = F.adaptive_avg_pool2d(x, (4, 4))
                x = x.view(x.size(0), -1)  # (batch_size, 512)
                
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
        if not self.cuda_available or self.device_count == 0:
            print("GPU가 사용 불가능합니다.")
            return
        
        print(f"🎯 {self.device_count}개 GPU에서 {self.duration_minutes}분 동안 집약적 워크로드 시작")
        print("   DCGM_FI_PROF_GR_ENGINE_ACTIVE 메트릭에 나타날 때까지 기다려주세요...")
        print("   ⚠️  컨테이너 환경에서 안전 모드로 실행됩니다.")
        
        # 🔍 디버깅 정보 추가
        print(f"\n🔍 GPU 감지 결과:")
        try:
            total_detected = torch.cuda.device_count()
            print(f"  전체 감지된 GPU: {total_detected}")
        except Exception as e:
            print(f"  전체 감지된 GPU: 확인 불가 ({e})")
        
        print(f"  사용 가능한 GPU 개수: {self.device_count}")
        print(f"  사용 가능한 GPU 인덱스: {self.available_devices}")
        
        # 각 사용 가능한 GPU에서 단일 워크로드만 실행 (안전성 향상)
        for device_id in self.available_devices:
            print(f"\n🚀 GPU {device_id}에서 워크로드 시작...")
            
            # MIG 환경에서 더 집약적인 사용을 위해 GPU당 2개 워크로드 실행
            for workload_id in range(2):
                thread = threading.Thread(
                    target=self.continuous_gpu_workload,
                    args=(device_id, workload_id)
                )
                thread.daemon = True
                thread.start()
                self.workload_threads.append(thread)
                print(f"  워크로드 {workload_id} 시작됨")
                
                # 워크로드 간 시작 간격 (메모리 경합 방지)
                time.sleep(1)
            
            # GPU 간 시작 간격 (리소스 경합 방지)
            time.sleep(2)
        
        print(f"\n✅ 총 {len(self.workload_threads)}개 워크로드 스레드 시작됨")
        
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
                    for i in self.available_devices:
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
            for i in self.available_devices:
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
    
    # GPU 정보 출력 (안전한 감지)
    try:
        if torch.cuda.is_available():
            print(f"\n🎮 GPU 정보:")
            try:
                # CUDA 초기화 테스트
                torch.cuda.init()
                device_count = torch.cuda.device_count()
                print(f"  감지된 GPU 개수: {device_count}")
                
                for i in range(device_count):
                    try:
                        # 단계적 GPU 접근 테스트
                        props = torch.cuda.get_device_properties(i)
                        device_name = props.name
                        total_memory = props.total_memory / 1024**3
                        
                        # 메모리 할당 테스트
                        with torch.cuda.device(i):
                            test_tensor = torch.zeros(1, device=f'cuda:{i}')
                            torch.cuda.synchronize(device=i)
                            del test_tensor
                        
                        print(f"  GPU {i}: {device_name}")
                        print(f"    메모리: {total_memory:.1f} GB")
                        print(f"    멀티프로세서: {props.multi_processor_count}")
                        
                    except Exception as e:
                        print(f"  GPU {i}: 접근 불가 - {str(e)[:100]}...")
                        
            except Exception as e:
                print(f"  GPU 정보 조회 실패: {e}")
        else:
            print("\n❌ CUDA 사용 불가능")
    except Exception as e:
        print(f"\n❌ GPU 확인 중 오류: {e}")
    
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
        
        # 안전한 GPU 상태 확인
        try:
            if hasattr(gpu_workload, 'cuda_available') and gpu_workload.cuda_available:
                if hasattr(gpu_workload, 'available_devices') and gpu_workload.available_devices:
                    for i in gpu_workload.available_devices:
                        gpu_utilization = final_usage.get(f'gpu_{i}_utilization', 0)
                        gpu_memory = final_usage.get(f'gpu_{i}_memory_used', 0)
                        print(f"  GPU {i} 사용률: {gpu_utilization}%")
                        print(f"  GPU {i} 메모리: {gpu_memory:.2f} GB")
                else:
                    print("  사용 가능한 GPU 없음")
            else:
                print("  GPU 사용 불가능")
        except Exception as e:
            print(f"  GPU 상태 확인 중 오류: {e}")
        
        # 안전한 GPU 메모리 정리
        print("\n🧹 GPU 메모리 정리 중...")
        try:
            if hasattr(gpu_workload, 'cuda_available') and gpu_workload.cuda_available:
                if hasattr(gpu_workload, 'available_devices') and gpu_workload.available_devices:
                    # 각 사용 가능한 GPU에서 개별적으로 정리
                    for device_id in gpu_workload.available_devices:
                        try:
                            with torch.cuda.device(device_id):
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize(device=device_id)
                            print(f"  GPU {device_id} 메모리 정리 완료")
                        except Exception as e:
                            print(f"  GPU {device_id} 메모리 정리 실패: {e}")
                else:
                    print("  정리할 GPU 없음")
            else:
                print("  GPU 정리 불가능")
        except Exception as e:
            print(f"  GPU 메모리 정리 중 오류: {e}")
        
        total_duration = time.time() - start_time
        print(f"\n🎯 PyTorch GPU 사용률 집약적 테스트 완료!")
        print(f"   총 실행 시간: {total_duration:.1f}초")
        print("=" * 60)


if __name__ == "__main__":
    main() 