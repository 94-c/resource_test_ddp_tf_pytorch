"""
TensorFlow CPU 집약적 테스트
이 테스트는 CPU 사용률을 최대화하여 CPU 리소스 모니터링을 테스트합니다.
"""

import tensorflow as tf
import numpy as np
import time
import argparse
import sys
import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.resource_monitor import ResourceMonitor, print_system_info


def setup_tensorflow_cpu():
    """TensorFlow CPU 설정"""
    # CPU만 사용하도록 설정
    tf.config.set_visible_devices([], 'GPU')
    
    # CPU 스레드 수 설정
    tf.config.threading.set_intra_op_parallelism_threads(mp.cpu_count())
    tf.config.threading.set_inter_op_parallelism_threads(mp.cpu_count())
    
    print(f"TensorFlow CPU 설정 완료 - 스레드 수: {mp.cpu_count()}")


def cpu_intensive_matrix_operations(num_iterations=200, matrix_size=2000):
    """CPU 집약적 행렬 연산"""
    print(f"TensorFlow CPU 집약적 행렬 연산 (반복: {num_iterations}, 크기: {matrix_size})")
    
    results = []
    
    for i in range(num_iterations):
        # 랜덤 행렬 생성
        a = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
        b = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
        
        # 행렬 곱셈
        c = tf.matmul(a, b)
        
        # 고유값 분해 (매우 CPU 집약적)
        if i % 10 == 0:
            try:
                eigenvalues = tf.linalg.eigvals(c[:500, :500])
                results.append(tf.reduce_mean(tf.math.real(eigenvalues)).numpy())
            except:
                pass
        
        # SVD 분해
        if i % 5 == 0:
            try:
                s, u, v = tf.linalg.svd(c[:300, :300])
                results.append(tf.reduce_mean(s).numpy())
            except:
                pass
        
        # 복잡한 수학 연산
        d = tf.sin(c) + tf.cos(c) + tf.exp(tf.clip_by_value(c, -5, 5))
        results.append(tf.reduce_mean(d).numpy())
        
        # 역행렬 계산
        try:
            c_inv = tf.linalg.inv(c + tf.eye(matrix_size) * 1e-5)
            results.append(tf.linalg.trace(c_inv).numpy())
        except:
            pass
        
        if i % 20 == 0:
            print(f"  반복 {i}/{num_iterations} 완료")
    
    return np.mean(results)


def cpu_parallel_tensorflow_ops(num_threads=None, operations_per_thread=100):
    """병렬 TensorFlow 연산"""
    if num_threads is None:
        num_threads = mp.cpu_count()
    
    print(f"병렬 TensorFlow 연산 ({num_threads} 스레드)")
    
    def worker_function(thread_id):
        """각 스레드에서 실행할 TensorFlow 연산"""
        results = []
        
        for i in range(operations_per_thread):
            # 텐서 생성
            size = 1000
            a = tf.random.normal([size, size])
            b = tf.random.normal([size, size])
            
            # 행렬 연산
            c = tf.matmul(a, b)
            d = tf.matmul(c, tf.transpose(a))
            
            # 통계 계산
            result = {
                'mean': tf.reduce_mean(d).numpy(),
                'std': tf.math.reduce_std(d).numpy(),
                'max': tf.reduce_max(d).numpy(),
                'min': tf.reduce_min(d).numpy()
            }
            results.append(result)
        
        return f"스레드 {thread_id} 완료: {len(results)} 연산"
    
    # 스레드 풀 실행
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_function, i) for i in range(num_threads)]
        results = [future.result() for future in futures]
    
    return results


def neural_network_cpu_training(epochs=40, batch_size=512):
    """CPU에서 신경망 학습"""
    print(f"TensorFlow CPU 신경망 학습 (에포크: {epochs}, 배치: {batch_size})")
    
    # 복잡한 신경망 모델 정의
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2048, activation='relu', input_shape=(784,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 가짜 데이터 생성
    def generate_batch():
        x = tf.random.normal([batch_size, 784])
        y = tf.random.uniform([batch_size], 0, 10, dtype=tf.int32)
        return x, y
    
    # 학습 루프
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch in range(10):  # 에포크당 10 배치
            x, y = generate_batch()
            
            # 그래디언트 테이프를 사용한 수동 학습
            with tf.GradientTape() as tape:
                predictions = model(x, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
                loss = tf.reduce_mean(loss)
            
            # 그래디언트 계산 및 적용
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_losses.append(loss.numpy())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"  에포크 {epoch}/{epochs}, 평균 손실: {avg_loss:.4f}")
    
    return losses


def convolution_cpu_intensive(num_iterations=100, batch_size=16):
    """CPU 집약적 컨볼루션 연산"""
    print(f"TensorFlow CPU 컨볼루션 연산 (반복: {num_iterations}, 배치: {batch_size})")
    
    # 컨볼루션 모델 생성
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 학습 루프
    for i in range(num_iterations):
        # 랜덤 이미지 데이터 생성
        x = tf.random.normal([batch_size, 224, 224, 3])
        y = tf.random.uniform([batch_size], 0, 10, dtype=tf.int32)
        
        # 학습 스텝
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
            loss = tf.reduce_mean(loss)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        if i % 20 == 0:
            print(f"  컨볼루션 반복 {i}/{num_iterations}, 손실: {loss.numpy():.4f}")


def mathematical_intensive_operations(num_iterations=500):
    """수학적 집약적 연산"""
    print(f"TensorFlow 수학적 집약적 연산 (반복: {num_iterations})")
    
    for i in range(num_iterations):
        # 대용량 텐서 생성
        size = 2000
        a = tf.random.normal([size, size])
        
        # 복잡한 수학 연산
        b = tf.sin(a) + tf.cos(a)
        c = tf.exp(tf.clip_by_value(b, -5, 5))
        d = tf.math.log(tf.abs(c) + 1e-8)
        e = tf.sqrt(tf.abs(d) + 1e-8)
        f = tf.pow(tf.abs(e), 0.5)
        
        # 행렬 연산
        g = tf.matmul(f, tf.transpose(f))
        
        # 통계 연산
        mean_val = tf.reduce_mean(g)
        std_val = tf.math.reduce_std(g)
        
        # 소팅 연산
        if i % 20 == 0:
            sorted_vals = tf.sort(tf.reshape(g, [-1]))
        
        # 조건부 연산
        mask = g > mean_val
        filtered = tf.where(mask, g, tf.zeros_like(g))
        
        if i % 50 == 0:
            print(f"  수학 연산 반복 {i}/{num_iterations}")


def main():
    parser = argparse.ArgumentParser(description='TensorFlow CPU 집약적 테스트')
    parser.add_argument('--duration', type=int, default=120, help='테스트 지속 시간 (초)')
    parser.add_argument('--matrix-iterations', type=int, default=100, help='행렬 연산 반복 횟수')
    parser.add_argument('--matrix-size', type=int, default=1500, help='행렬 크기')
    parser.add_argument('--epochs', type=int, default=30, help='신경망 학습 에포크')
    parser.add_argument('--skip-parallel', action='store_true', help='병렬 처리 테스트 건너뛰기')
    parser.add_argument('--skip-training', action='store_true', help='신경망 학습 테스트 건너뛰기')
    parser.add_argument('--skip-conv', action='store_true', help='컨볼루션 테스트 건너뛰기')
    
    args = parser.parse_args()
    
    # TensorFlow CPU 설정
    setup_tensorflow_cpu()
    
    # 시스템 정보 출력
    print_system_info()
    
    # TensorFlow 정보 출력
    print(f"\nTensorFlow 버전: {tf.__version__}")
    print(f"사용 가능한 CPU: {len(tf.config.list_physical_devices('CPU'))}")
    
    # 리소스 모니터링 시작
    monitor = ResourceMonitor(interval=0.5)
    monitor.start_monitoring()
    
    print(f"\n🔥 TensorFlow CPU 집약적 테스트 시작 (지속 시간: {args.duration}초)")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # 테스트 실행
        test_results = {}
        
        # 1. 행렬 연산 테스트
        print("\n1. 행렬 연산 테스트")
        result = cpu_intensive_matrix_operations(
            num_iterations=args.matrix_iterations,
            matrix_size=args.matrix_size
        )
        test_results['matrix_operations'] = result
        
        # 2. 병렬 처리 테스트
        if not args.skip_parallel:
            print("\n2. 병렬 처리 테스트")
            result = cpu_parallel_tensorflow_ops()
            test_results['parallel_processing'] = result
        
        # 3. 신경망 학습 테스트
        if not args.skip_training:
            print("\n3. 신경망 학습 테스트")
            result = neural_network_cpu_training(epochs=args.epochs)
            test_results['neural_network'] = result
        
        # 4. 컨볼루션 테스트
        if not args.skip_conv:
            print("\n4. 컨볼루션 테스트")
            convolution_cpu_intensive(num_iterations=50)
            test_results['convolution'] = True
        
        # 5. 수학적 집약적 연산
        print("\n5. 수학적 집약적 연산")
        mathematical_intensive_operations(num_iterations=200)
        test_results['mathematical_ops'] = True
        
        # 남은 시간 동안 추가 행렬 연산 수행
        elapsed_time = time.time() - start_time
        remaining_time = args.duration - elapsed_time
        
        if remaining_time > 10:
            print(f"\n6. 추가 행렬 연산 ({remaining_time:.1f}초 동안)")
            extra_iterations = max(20, int(remaining_time / 3))
            result = cpu_intensive_matrix_operations(
                num_iterations=extra_iterations,
                matrix_size=1000
            )
            test_results['extra_operations'] = result
        
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
        print(f"CPU: {final_usage.get('cpu_percent', 0):.1f}%")
        print(f"Memory: {final_usage.get('memory_percent', 0):.1f}%")
        
        print("\n🎯 TensorFlow CPU 집약적 테스트 완료!")
        print("=" * 60)


if __name__ == "__main__":
    main() 