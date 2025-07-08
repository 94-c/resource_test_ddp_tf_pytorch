"""
TensorFlow GPU 사용률 집약적 테스트 (DCGM 메트릭 대응)
이 테스트는 GPU 사용률을 최대화하여 DCGM_FI_PROF_GR_ENGINE_ACTIVE 메트릭에 나타나도록 합니다.
"""

import tensorflow as tf
import numpy as np
import time
import argparse
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import signal

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.resource_monitor import ResourceMonitor, print_system_info


class IntensiveTensorFlowGPUWorkload:
    """DCGM 메트릭에 나타나도록 하는 집약적 TensorFlow GPU 워크로드"""
    
    def __init__(self, duration_minutes=10):
        self.duration_minutes = duration_minutes
        self.duration_seconds = duration_minutes * 60
        self.stop_event = threading.Event()
        self.workload_threads = []
        
        # GPU 설정
        self.gpus = tf.config.list_physical_devices('GPU')
        if self.gpus:
            try:
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ TensorFlow GPU 설정 완료: {len(self.gpus)}개 GPU 사용 가능")
                for i, gpu in enumerate(self.gpus):
                    print(f"  GPU {i}: {gpu.name}")
            except RuntimeError as e:
                print(f"❌ GPU 설정 오류: {e}")
                self.gpus = []
        else:
            print("⚠️  GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
    
    def create_intensive_model(self, device_name):
        """GPU 집약적 모델 생성"""
        with tf.device(device_name):
            # 매우 큰 모델로 GPU 사용률 최대화
            model = tf.keras.Sequential([
                # 대용량 컨볼루션 레이어들
                tf.keras.layers.Conv2D(128, (7, 7), activation='relu', input_shape=(512, 512, 3), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                
                tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                
                tf.keras.layers.Conv2D(2048, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                
                # 업샘플링 레이어들
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                
                # 글로벌 평균 풀링 및 Dense 레이어들
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(2048, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(2048, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1000, activation='relu'),
                tf.keras.layers.Dense(100, activation='softmax')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
    
    def continuous_gpu_workload(self, gpu_id, workload_id):
        """지속적인 GPU 워크로드 실행"""
        print(f"🚀 GPU {gpu_id} 워크로드 {workload_id} 시작")
        
        try:
            device_name = f'/GPU:{gpu_id}'
            model = self.create_intensive_model(device_name)
            
            # 큰 배치 크기로 GPU 사용률 최대화
            batch_size = 16
            start_time = time.time()
            iteration = 0
            
            with tf.device(device_name):
                while not self.stop_event.is_set() and (time.time() - start_time) < self.duration_seconds:
                    try:
                        # 큰 이미지 데이터 생성
                        X = tf.random.normal([batch_size, 512, 512, 3])
                        y = tf.random.uniform([batch_size], 0, 100, dtype=tf.int32)
                        
                        # 학습 스텝
                        with tf.GradientTape() as tape:
                            predictions = model(X, training=True)
                            loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
                            loss = tf.reduce_mean(loss)
                        
                        gradients = tape.gradient(loss, model.trainable_variables)
                        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                        
                        # 추가 계산 집약적 연산들
                        if iteration % 5 == 0:
                            # 대용량 행렬 연산
                            matrix_a = tf.random.normal([2048, 2048])
                            matrix_b = tf.random.normal([2048, 2048])
                            result = tf.matmul(matrix_a, matrix_b)
                            
                            # 고유값 분해 (매우 집약적)
                            try:
                                eigenvalues = tf.linalg.eigvals(result[:1024, :1024])
                            except:
                                pass
                            
                            # FFT 연산
                            signal_data = tf.random.normal([8192])
                            fft_result = tf.signal.fft(tf.cast(signal_data, tf.complex64))
                            ifft_result = tf.signal.ifft(fft_result)
                            
                            # 2D FFT
                            image_data = tf.random.normal([512, 512])
                            fft_2d = tf.signal.fft2d(tf.cast(image_data, tf.complex64))
                            ifft_2d = tf.signal.ifft2d(fft_2d)
                        
                        # 메모리 압박 방지를 위한 주기적 정리
                        if iteration % 20 == 0:
                            tf.keras.backend.clear_session()
                            elapsed = time.time() - start_time
                            remaining = self.duration_seconds - elapsed
                            print(f"  GPU {gpu_id} 워크로드 {workload_id}: {iteration} 반복 완료, "
                                  f"남은 시간: {remaining:.1f}초, Loss: {loss.numpy():.4f}")
                        
                        iteration += 1
                        
                    except tf.errors.ResourceExhaustedError:
                        print(f"  GPU {gpu_id} 메모리 부족, 배치 크기 줄임")
                        batch_size = max(1, batch_size // 2)
                        tf.keras.backend.clear_session()
                        model = self.create_intensive_model(device_name)
                        continue
                    except Exception as e:
                        print(f"  GPU {gpu_id} 워크로드 오류: {e}")
                        continue
            
            print(f"✅ GPU {gpu_id} 워크로드 {workload_id} 완료 ({iteration} 반복)")
            
        except Exception as e:
            print(f"❌ GPU {gpu_id} 워크로드 {workload_id} 실패: {e}")
    
    def start_workloads(self):
        """모든 GPU에서 워크로드 시작"""
        if not self.gpus:
            print("GPU가 사용 불가능합니다.")
            return
        
        print(f"🎯 {len(self.gpus)}개 GPU에서 {self.duration_minutes}분 동안 집약적 워크로드 시작")
        print("   DCGM_FI_PROF_GR_ENGINE_ACTIVE 메트릭에 나타날 때까지 기다려주세요...")
        
        # 각 GPU에서 여러 워크로드 실행
        for gpu_id in range(len(self.gpus)):
            # GPU당 2개의 워크로드 스레드 실행
            for workload_id in range(2):
                thread = threading.Thread(
                    target=self.continuous_gpu_workload,
                    args=(gpu_id, workload_id)
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
        
        print(f"🏁 모든 GPU 워크로드 완료")


def main():
    parser = argparse.ArgumentParser(description='TensorFlow GPU 사용률 집약적 테스트 (DCGM 메트릭 대응)')
    parser.add_argument('--duration', type=int, default=600, help='테스트 지속 시간 (초)')
    
    args = parser.parse_args()
    
    # 시스템 정보 출력
    print_system_info()
    
    # GPU 정보 출력
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n🎮 TensorFlow GPU 정보:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    
    # 리소스 모니터링 시작
    monitor = ResourceMonitor(interval=1)
    monitor.start_monitoring()
    
    duration_minutes = max(1, args.duration // 60)
    print(f"\n🚀 TensorFlow GPU 사용률 집약적 테스트 시작 ({duration_minutes}분)")
    print("   DCGM_FI_PROF_GR_ENGINE_ACTIVE 메트릭에 나타날 때까지 기다려주세요...")
    print("=" * 60)
    
    # GPU 사용률 최대화 객체 생성
    gpu_workload = IntensiveTensorFlowGPUWorkload(duration_minutes=duration_minutes)
    
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
        if gpus:
            for i in range(len(gpus)):
                gpu_utilization = final_usage.get(f'gpu_{i}_utilization', 0)
                gpu_memory = final_usage.get(f'gpu_{i}_memory_used', 0)
                print(f"  GPU {i} 사용률: {gpu_utilization}%")
                print(f"  GPU {i} 메모리: {gpu_memory:.2f} GB")
        
        # 메모리 정리
        print("\n🧹 메모리 정리 중...")
        tf.keras.backend.clear_session()
        
        total_duration = time.time() - start_time
        print(f"\n🎯 TensorFlow GPU 사용률 집약적 테스트 완료!")
        print(f"   총 실행 시간: {total_duration:.1f}초")
        print("=" * 60)


if __name__ == "__main__":
    main() 