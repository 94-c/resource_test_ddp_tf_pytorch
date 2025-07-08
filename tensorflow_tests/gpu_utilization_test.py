"""
TensorFlow GPU 사용률 집약적 테스트
이 테스트는 GPU 사용률을 최대화하여 GPU 활용도 모니터링을 테스트합니다.
"""

import tensorflow as tf
import numpy as np
import time
import argparse
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.resource_monitor import ResourceMonitor, print_system_info


def setup_tensorflow_gpu():
    """TensorFlow GPU 설정"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU 설정 완료: {len(gpus)}개 GPU 사용 가능")
            return True
        except RuntimeError as e:
            print(f"GPU 설정 오류: {e}")
            return False
    else:
        print("⚠️  GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        return False


def parallel_gpu_matrix_operations(num_operations=1000, matrix_size=2000):
    """병렬 GPU 행렬 연산"""
    if not tf.config.list_physical_devices('GPU'):
        print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
        return
    
    print(f"병렬 GPU 행렬 연산: {num_operations} 연산, 크기: {matrix_size}")
    
    def matrix_worker(worker_id, operations_per_worker):
        """각 워커에서 실행할 행렬 연산"""
        with tf.device('/GPU:0'):
            for i in range(operations_per_worker):
                # 다양한 행렬 연산
                a = tf.random.normal([matrix_size, matrix_size])
                b = tf.random.normal([matrix_size, matrix_size])
                
                # 행렬 곱셈
                c = tf.matmul(a, b)
                
                # 고유값 분해 (계산 집약적)
                if i % 10 == 0:
                    try:
                        eigenvalues = tf.linalg.eigvals(c[:500, :500])
                    except:
                        pass
                
                # 삼각함수 연산
                d = tf.sin(c) + tf.cos(c) + tf.tan(tf.clip_by_value(c, -1, 1))
                
                # 지수 및 로그 연산
                e = tf.exp(tf.clip_by_value(d, -5, 5))
                f = tf.math.log(tf.abs(e) + 1e-8)
                
                # 역행렬 계산
                try:
                    g = tf.linalg.inv(f + tf.eye(matrix_size) * 1e-3)
                except:
                    g = f
    
    # 병렬 실행
    num_workers = 4
    operations_per_worker = num_operations // num_workers
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            future = executor.submit(matrix_worker, i, operations_per_worker)
            futures.append(future)
        
        # 모든 작업 완료 대기
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"워커 작업 중 오류: {e}")


def convolution_gpu_intensive(num_iterations=200, batch_size=16):
    """GPU 집약적 컨볼루션 연산"""
    if not tf.config.list_physical_devices('GPU'):
        print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
        return
    
    print(f"GPU 컨볼루션 집약적 테스트: {num_iterations} 반복, 배치: {batch_size}")
    
    with tf.device('/GPU:0'):
        # 컨볼루션 네트워크 생성
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        for i in range(num_iterations):
            try:
                # 랜덤 이미지 데이터 생성
                X = tf.random.normal([batch_size, 224, 224, 3])
                y = tf.random.uniform([batch_size], 0, 10, dtype=tf.int32)
                
                # 학습 스텝
                with tf.GradientTape() as tape:
                    predictions = model(X, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
                    loss = tf.reduce_mean(loss)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                if i % 20 == 0:
                    print(f"  컨볼루션 반복 {i}/{num_iterations}")
                    
            except tf.errors.ResourceExhaustedError:
                print(f"  GPU 메모리 부족으로 반복 {i}에서 중단")
                break


def fft_gpu_intensive(num_iterations=500, signal_size=8192):
    """GPU 집약적 FFT 연산"""
    if not tf.config.list_physical_devices('GPU'):
        print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
        return
    
    print(f"GPU FFT 집약적 테스트: {num_iterations} 반복, 신호 크기: {signal_size}")
    
    with tf.device('/GPU:0'):
        for i in range(num_iterations):
            # 복잡한 신호 생성
            real_signal = tf.random.normal([signal_size])
            imag_signal = tf.random.normal([signal_size])
            complex_signal = tf.complex(real_signal, imag_signal)
            
            # FFT 연산
            fft_result = tf.signal.fft(complex_signal)
            
            # 역 FFT
            ifft_result = tf.signal.ifft(fft_result)
            
            # 2D FFT (이미지 처리 시뮬레이션)
            if i % 10 == 0:
                image_size = int(np.sqrt(signal_size))
                if image_size * image_size <= signal_size:
                    image = tf.reshape(real_signal[:image_size*image_size], [image_size, image_size])
                    image_complex = tf.cast(image, tf.complex64)
                    fft_2d = tf.signal.fft2d(image_complex)
                    ifft_2d = tf.signal.ifft2d(fft_2d)
            
            # 스펙트럼 분석
            power_spectrum = tf.abs(fft_result) ** 2
            
            if i % 50 == 0:
                print(f"  FFT 반복 {i}/{num_iterations}")


def compute_intensive_gpu_operations(num_iterations=300):
    """GPU 계산 집약적 연산"""
    if not tf.config.list_physical_devices('GPU'):
        print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
        return
    
    print(f"GPU 계산 집약적 연산: {num_iterations} 반복")
    
    with tf.device('/GPU:0'):
        for i in range(num_iterations):
            # 대용량 텐서 생성
            size = 2000
            a = tf.random.normal([size, size])
            b = tf.random.normal([size, size])
            
            # 복잡한 수학 연산들
            c = tf.matmul(a, b)
            d = tf.sin(c) * tf.cos(c) + tf.tan(tf.clip_by_value(c, -1, 1))
            e = tf.exp(tf.clip_by_value(d, -5, 5))
            f = tf.math.log(tf.abs(e) + 1e-8)
            g = tf.sqrt(tf.abs(f) + 1e-8)
            h = tf.pow(tf.abs(g), 0.5)
            
            # 통계적 연산
            mean_val = tf.reduce_mean(h)
            std_val = tf.math.reduce_std(h)
            max_val = tf.reduce_max(h)
            min_val = tf.reduce_min(h)
            
            # 소팅 연산
            if i % 20 == 0:
                sorted_vals = tf.sort(tf.reshape(h, [-1]))
            
            # 조건부 연산
            mask = h > mean_val
            filtered = tf.where(mask, h, tf.zeros_like(h))
            
            if i % 30 == 0:
                print(f"  계산 집약적 반복 {i}/{num_iterations}")


def mixed_precision_gpu_test(num_iterations=200):
    """GPU 혼합 정밀도 테스트"""
    if not tf.config.list_physical_devices('GPU'):
        print("GPU가 사용 불가능합니다. 테스트를 건너뜁니다.")
        return
    
    print(f"GPU 혼합 정밀도 테스트: {num_iterations} 반복")
    
    with tf.device('/GPU:0'):
        # 혼합 정밀도 정책 설정
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        # 모델 생성
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2048, activation='relu', input_shape=(1024,)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax', dtype='float32')  # 출력은 float32
        ])
        
        optimizer = tf.keras.optimizers.Adam()
        
        for i in range(num_iterations):
            try:
                # 배치 데이터 생성
                X = tf.random.normal([64, 1024])
                y = tf.random.uniform([64], 0, 10, dtype=tf.int32)
                
                with tf.GradientTape() as tape:
                    predictions = model(X, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
                    loss = tf.reduce_mean(loss)
                    
                    # 손실 스케일링
                    scaled_loss = optimizer.get_scaled_loss(loss)
                
                # 그래디언트 계산
                scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
                gradients = optimizer.get_unscaled_gradients(scaled_gradients)
                
                # 그래디언트 적용
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                if i % 20 == 0:
                    print(f"  혼합 정밀도 반복 {i}/{num_iterations}")
                    
            except tf.errors.ResourceExhaustedError:
                print(f"  GPU 메모리 부족으로 반복 {i}에서 중단")
                break
        
        # 정책 리셋
        tf.keras.mixed_precision.set_global_policy('float32')


def main():
    parser = argparse.ArgumentParser(description='TensorFlow GPU 사용률 집약적 테스트')
    parser.add_argument('--duration', type=int, default=240, help='테스트 지속 시간 (초)')
    parser.add_argument('--matrix-ops', type=int, default=400, help='행렬 연산 횟수')
    parser.add_argument('--conv-iterations', type=int, default=100, help='컨볼루션 반복 횟수')
    parser.add_argument('--skip-conv', action='store_true', help='컨볼루션 테스트 건너뛰기')
    parser.add_argument('--skip-fft', action='store_true', help='FFT 테스트 건너뛰기')
    parser.add_argument('--skip-mixed-precision', action='store_true', help='혼합 정밀도 테스트 건너뛰기')
    
    args = parser.parse_args()
    
    # TensorFlow GPU 설정
    gpu_available = setup_tensorflow_gpu()
    
    # 시스템 정보 출력
    print_system_info()
    
    # TensorFlow 정보
    print(f"\nTensorFlow 버전: {tf.__version__}")
    if gpu_available:
        gpus = tf.config.list_physical_devices('GPU')
        print(f"사용 가능한 GPU: {len(gpus)}")
    
    # 리소스 모니터링 시작
    monitor = ResourceMonitor(interval=0.5)
    monitor.start_monitoring()
    
    print(f"\n🚀 TensorFlow GPU 사용률 집약적 테스트 시작 (지속 시간: {args.duration}초)")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 테스트 실행
        
        # 1. 병렬 행렬 연산
        print("\n1. 병렬 GPU 행렬 연산")
        parallel_gpu_matrix_operations(
            num_operations=args.matrix_ops,
            matrix_size=1500
        )
        
        # 2. 컨볼루션 집약적 테스트
        if not args.skip_conv:
            print("\n2. GPU 컨볼루션 집약적 테스트")
            convolution_gpu_intensive(
                num_iterations=args.conv_iterations,
                batch_size=8
            )
        
        # 3. FFT 집약적 테스트
        if not args.skip_fft:
            print("\n3. GPU FFT 집약적 테스트")
            fft_gpu_intensive(
                num_iterations=200,
                signal_size=4096
            )
        
        # 4. 계산 집약적 연산
        print("\n4. GPU 계산 집약적 연산")
        compute_intensive_gpu_operations(num_iterations=150)
        
        # 5. 혼합 정밀도 테스트
        if not args.skip_mixed_precision:
            print("\n5. GPU 혼합 정밀도 테스트")
            mixed_precision_gpu_test(num_iterations=100)
        
        # 남은 시간 동안 추가 연산 수행
        elapsed_time = time.time() - start_time
        remaining_time = args.duration - elapsed_time
        
        if remaining_time > 20:
            print(f"\n6. 추가 병렬 연산 ({remaining_time:.1f}초 동안)")
            extra_operations = max(100, int(remaining_time * 3))
            parallel_gpu_matrix_operations(
                num_operations=extra_operations,
                matrix_size=1200
            )
        
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
        if gpu_available:
            for i in range(len(tf.config.list_physical_devices('GPU'))):
                gpu_utilization = final_usage.get(f'gpu_{i}_utilization', 0)
                print(f"GPU {i} 사용률: {gpu_utilization}%")
        
        print("\n🎯 TensorFlow GPU 사용률 집약적 테스트 완료!")
        print("=" * 60)


if __name__ == "__main__":
    main() 