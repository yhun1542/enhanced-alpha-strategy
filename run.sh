#!/bin/bash
# Enhanced Alpha Strategy - 실행 스크립트

echo "======================================"
echo "Enhanced Alpha Strategy v5.0"
echo "======================================"
echo ""

# API 키 확인
if [ -z "$POLYGON_API_KEY" ]; then
    echo "❌ Error: POLYGON_API_KEY 환경변수가 설정되지 않았습니다."
    echo ""
    echo "사용법:"
    echo "  export POLYGON_API_KEY='your_api_key_here'"
    echo "  ./run.sh"
    exit 1
fi

echo "✅ Polygon API Key: ${POLYGON_API_KEY:0:10}..."
echo ""

# Python 버전 확인
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ Error: Python 3.x가 설치되지 않았습니다."
    exit 1
fi

echo "✅ Python: $($PYTHON_CMD --version)"
echo ""

# 패키지 확인
echo "📦 필수 패키지 확인 중..."
$PYTHON_CMD -c "import pandas, numpy, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  필수 패키지가 설치되지 않았습니다."
    echo "설치 중..."
    pip3 install -r requirements.txt -q
fi
echo "✅ 패키지 확인 완료"
echo ""

# 백테스트 실행
echo "🚀 백테스트 실행 중..."
echo ""
$PYTHON_CMD enhanced_alpha_strategy_v5.py

# 결과 확인
if [ -f "enhanced_alpha_results.json" ]; then
    echo ""
    echo "======================================"
    echo "✅ 백테스트 완료!"
    echo "======================================"
    echo ""
    echo "결과 파일:"
    echo "  - enhanced_alpha_results.json"
    echo ""
else
    echo ""
    echo "❌ 백테스트 실패"
    exit 1
fi
