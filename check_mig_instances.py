#!/usr/bin/env python3
"""
MIG 인스턴스 확인 스크립트
단일 GPU에서 여러 MIG 인스턴스 상황을 확인합니다.
"""

import subprocess
import json
import re

def check_mig_instances():
    """MIG 인스턴스 확인"""
    print("🔍 MIG 인스턴스 확인")
    print("=" * 50)
    
    try:
        # nvidia-smi로 MIG 인스턴스 확인
        result = subprocess.run([
            'nvidia-smi', 'mig', '-lgip'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ MIG 인스턴스 목록:")
            print(result.stdout)
        else:
            print("❌ MIG가 활성화되지 않았거나 사용할 수 없습니다")
            
    except FileNotFoundError:
        print("❌ nvidia-smi를 찾을 수 없습니다")

def check_gpu_utilization():
    """GPU 사용률 확인"""
    print("\n🎯 GPU 사용률 확인")
    print("=" * 50)
    
    try:
        # nvidia-smi로 GPU 사용률 확인
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 5:
                        gpu_id, name, util, mem_used, mem_total = parts[:5]
                        print(f"GPU {gpu_id}: {name}")
                        print(f"  사용률: {util}%")
                        print(f"  메모리: {mem_used}MB / {mem_total}MB")
                        print()
        else:
            print("❌ GPU 정보를 가져올 수 없습니다")
            
    except FileNotFoundError:
        print("❌ nvidia-smi를 찾을 수 없습니다")

def simulate_mig_scenario():
    """MIG 시나리오 시뮬레이션"""
    print("\n💡 MIG 시나리오 예시")
    print("=" * 50)
    
    print("🖥️  시나리오: A100 80GB GPU 1개에서 여러 MIG 인스턴스")
    print()
    print("1. MIG 인스턴스 생성:")
    print("   sudo nvidia-smi mig -cgi 1g.10gb,1g.10gb,2g.20gb,3g.20gb")
    print()
    print("2. 결과: 물리적 GPU 1개 → 논리적 인스턴스 4개")
    print("   ├── MIG 0: 1g.10gb (슬라이스 비중: 0.143)")
    print("   ├── MIG 1: 1g.10gb (슬라이스 비중: 0.143)")
    print("   ├── MIG 2: 2g.20gb (슬라이스 비중: 0.286)")
    print("   └── MIG 3: 3g.20gb (슬라이스 비중: 0.429)")
    print()
    print("3. Pod 할당:")
    print("   ├── Pod A: MIG 0 사용 (1g.10gb)")
    print("   ├── Pod B: MIG 1 사용 (1g.10gb)")
    print("   ├── Pod C: MIG 2 사용 (2g.20gb)")
    print("   └── Pod D: MIG 3 사용 (3g.20gb)")
    print()
    print("4. 가중평균 계산:")
    print("   전체 GPU 사용률 = (Pod A × 0.143 + Pod B × 0.143 + Pod C × 0.286 + Pod D × 0.429)")
    print("                    / (0.143 + 0.143 + 0.286 + 0.429)")

def check_kubernetes_mig():
    """Kubernetes MIG 설정 확인"""
    print("\n🚢 Kubernetes MIG 설정 확인")
    print("=" * 50)
    
    try:
        # kubectl로 노드 정보 확인
        result = subprocess.run([
            'kubectl', 'get', 'nodes', '-o', 'json'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            nodes_data = json.loads(result.stdout)
            
            for node in nodes_data['items']:
                node_name = node['metadata']['name']
                
                # MIG 자원 확인
                if 'status' in node and 'capacity' in node['status']:
                    capacity = node['status']['capacity']
                    mig_resources = {k: v for k, v in capacity.items() if 'nvidia.com/mig' in k}
                    
                    if mig_resources:
                        print(f"노드: {node_name}")
                        print("MIG 자원:")
                        for resource, count in mig_resources.items():
                            print(f"  {resource}: {count}")
                        print()
                    else:
                        print(f"노드: {node_name} - MIG 자원 없음")
        else:
            print("❌ kubectl를 사용할 수 없습니다")
            
    except FileNotFoundError:
        print("❌ kubectl을 찾을 수 없습니다")
    except json.JSONDecodeError:
        print("❌ kubectl 응답을 파싱할 수 없습니다")

def main():
    """메인 함수"""
    print("🔧 MIG 인스턴스 분석 도구")
    print("=" * 60)
    print("단일 GPU에서 여러 MIG 인스턴스 확인")
    print()
    
    check_mig_instances()
    check_gpu_utilization()
    simulate_mig_scenario()
    check_kubernetes_mig()
    
    print("\n💡 결론")
    print("=" * 50)
    print("✅ 하나의 GPU에서도 여러 MIG 인스턴스 생성 가능")
    print("✅ 각 Pod는 일반적으로 하나의 MIG 인스턴스 사용")
    print("✅ 가중평균은 같은 GPU의 여러 MIG 인스턴스 사용률 종합")
    print("✅ 멀티 GPU 환경이 필수가 아님")

if __name__ == "__main__":
    main() 