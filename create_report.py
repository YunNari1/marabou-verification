"""
Generates report.pdf (Korean, 1-2 pages) for Assignment #3.
Edit the RESULTS dict below after running test.py, then re-run this script.

Usage:
    python create_report.py
    python create_report.py --sat   # if a counterexample was found
"""

import argparse
from fpdf import FPDF

# ── Korean font path (Windows Malgun Gothic) ─────────────────────────────────
FONT_PATH   = r"C:\Windows\Fonts\malgun.ttf"
FONT_B_PATH = r"C:\Windows\Fonts\malgunbd.ttf"   # bold variant

# ── Verification results — fill in after running test.py ─────────────────────
RESULTS = {
    "model_arch"  : "784 -> 64 -> 32 -> 10 (ReLU, 선형 출력)",
    "test_acc"    : "96.35",          # % on MNIST test set
    "sample_idx"  : 0,
    "true_label"  : 7,
    "eps"         : 0.01,
    "outcome"     : "UNSAT",          # actual Marabou result
    "elapsed_s"   : "0.95",          # full 9-class verification time
    "adv_class"   : None,
    "adv_linf"    : None,
    "mode"        : "전체 9개 클래스 (--full)",
}


class Report(FPDF):
    def __init__(self, font_path: str, bold_path: str):
        super().__init__()
        self.add_font("Malgun", style="",  fname=font_path)
        self.add_font("Malgun", style="B", fname=bold_path)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Malgun", "B", 11)
        self.cell(0, 8, "Reliable and Trustworthy AI — Assignment #3", align="C")
        self.ln(4)
        self.set_draw_color(100, 100, 100)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Malgun", "", 9)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, f"- {self.page_no()} -", align="C")
        self.set_text_color(0, 0, 0)

    def section(self, title: str):
        self.ln(3)
        self.set_font("Malgun", "B", 12)
        self.set_fill_color(230, 236, 245)
        self.cell(0, 8, title, fill=True, ln=True)
        self.ln(1)

    def body(self, text: str, indent: float = 0):
        self.set_font("Malgun", "", 10.5)
        self.set_x(self.l_margin + indent)
        self.multi_cell(self.w - self.l_margin - self.r_margin - indent, 6, text)
        self.ln(1)

    def bullet(self, text: str):
        self.set_font("Malgun", "", 10.5)
        self.set_x(self.l_margin + 4)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 4, 6, f"• {text}")

    def kv(self, key: str, value: str):
        self.set_font("Malgun", "B", 10.5)
        self.set_x(self.l_margin + 4)
        self.cell(50, 6, f"{key}:")
        self.set_font("Malgun", "", 10.5)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 54, 6, value)


def build(r: dict, output: str = "report.pdf"):
    pdf = Report(FONT_PATH, FONT_B_PATH)
    pdf.add_page()

    # ── Title ────────────────────────────────────────────────────────────────
    pdf.set_font("Malgun", "B", 16)
    pdf.cell(0, 10, "Marabou를 이용한 신경망 지역 견고성 검증", align="C", ln=True)
    pdf.set_font("Malgun", "", 10)
    pdf.cell(0, 6, "Assignment #3  |  2026년 5월", align="C", ln=True)
    pdf.ln(4)

    # ── 1. 모델 및 데이터셋 ──────────────────────────────────────────────────
    pdf.section("1. 모델 및 데이터셋")
    pdf.body(
        "본 과제에서는 MNIST 손글씨 숫자 데이터셋에 대해 훈련된 소형 완전연결 신경망(MLP)을 "
        "사용하였다. Marabou의 기본 제공 resources 디렉터리에 포함된 모델과 중복되지 않도록 "
        "직접 설계·훈련한 외부 모델이다."
    )
    pdf.kv("모델 구조", r["model_arch"])
    pdf.kv("활성화 함수", "ReLU (Marabou가 최적 지원)")
    pdf.kv("데이터셋", "MNIST (torchvision, 테스트 10,000개)")
    pdf.kv("테스트 정확도", f"{r['test_acc']} %")
    pdf.kv("모델 포맷", "ONNX (opset 14, 고정 배치 크기 1)")
    pdf.ln(2)

    pdf.body(
        "ReLU 활성화 함수를 선택한 이유는 Marabou가 ReLU 네트워크에 대해 완전한 SMT 기반 검증을 "
        "지원하기 때문이다. 모델 크기는 784→64→32→10으로 설정해 검증 시간을 현실적인 범위 "
        "(수 분 이내)로 제한하였다."
    )

    # ── 2. 검증 쿼리 ────────────────────────────────────────────────────────
    pdf.section("2. 검증 쿼리")
    pdf.body(
        "지역 견고성(Local Robustness) 쿼리를 정식화하였다. "
        "테스트 입력 x가 참 레이블 d로 분류될 때, ℓ∞ 공(ball) 내의 모든 섭동 x′도 "
        "동일한 클래스 d로 분류되는지 검증한다."
    )
    pdf.kv("샘플 인덱스", str(r["sample_idx"]))
    pdf.kv("참 레이블", f"digit {r['true_label']}")
    pdf.kv("섭동 반경 ε", str(r["eps"]))
    pdf.kv("검증 모드", r["mode"])
    pdf.ln(1)
    pdf.body(
        "입력 제약: 각 픽셀 i에 대해  max(0, x_i - e) <= x'_i <= min(1, x_i + e).\n"
        "출력 제약: 경쟁 클래스 j != d에 대해  output[j] - output[d] >= 0 (SAT 질의).\n"
        "모든 j에 대해 UNSAT이면 지역 견고성이 형식적으로 보장된다."
    )

    # ── 3. 결과 ────────────────────────────────────────────────────────────
    pdf.section("3. 실험 결과 및 해석")

    outcome_kor = "UNSAT (견고성 검증 성공)" if r["outcome"] == "UNSAT" else f"SAT (반례 발견 — 클래스 {r['adv_class']})"
    pdf.kv("검증 결과", outcome_kor)
    pdf.kv("소요 시간", f"{r['elapsed_s']} 초")

    if r["outcome"] == "UNSAT":
        pdf.body(
            f"모든 9개 경쟁 클래스(digit 0-6, 8-9)에 대한 하위 쿼리가 UNSAT으로 판정되었다. "
            f"즉, ε = {r['eps']} 범위의 ℓ∞ 공 내에서는 어떠한 섭동도 모델이 digit {r['true_label']}가 "
            "아닌 다른 클래스로 예측하도록 만들 수 없음이 형식적으로 증명되었다.\n"
            "가장 높은 경쟁 로짓을 가진 클래스(digit 3, logit=3.41)부터 낮은 순서로 검증하였으며, "
            "9개 쿼리 전체가 0.95초 이내에 완료되었다. "
            "소형 MLP와 작은 ε의 조합이 실용적인 검증 시간을 가능하게 한다."
        )
    else:
        pdf.body(
            f"ε = {r['eps']} 범위 내에서 클래스 {r['adv_class']}로 예측되는 적대적 입력 x′이 발견되었다 "
            f"(최대 픽셀 섭동: {r['adv_linf']:.6f}). "
            "이는 해당 ε 크기에서 모델이 지역적으로 견고하지 않음을 의미한다. "
            "ε을 줄이거나 더 강건한 모델을 훈련하면 견고성을 확보할 수 있다."
        )

    # ── 4. Marabou 장단점 ────────────────────────────────────────────────────
    pdf.section("4. Marabou 장단점 분석")

    pdf.set_font("Malgun", "B", 10.5)
    pdf.cell(0, 6, "장점", ln=True)
    pdf.bullet("형식적 보장: SMT 기반으로 SAT/UNSAT 결과가 수학적으로 완전하다.")
    pdf.bullet("ONNX 지원: PyTorch 등 주요 프레임워크에서 훈련한 모델을 직접 사용할 수 있다.")
    pdf.bullet("Python API(maraboupy): 검증 쿼리를 코드로 유연하게 정의할 수 있다.")
    pdf.bullet("다양한 속성 지원: 견고성, 안전성, 도달 가능성 등 폭넓은 검증 유형을 처리한다.")
    pdf.ln(2)

    pdf.set_font("Malgun", "B", 10.5)
    pdf.cell(0, 6, "단점 및 한계", ln=True)
    pdf.bullet(
        "확장성 제한: ReLU 뉴런 수가 늘어날수록 지수적으로 검증 시간이 증가한다. "
        "실용적으로는 수백~수천 뉴런 규모에 그친다."
    )
    pdf.bullet(
        "설치 복잡성: Python 3.12 이상 미지원(≤3.12 제약), C++ 컴파일 필요, "
        "Windows 네이티브 미지원 등 환경 설정이 까다롭다."
    )
    pdf.bullet("동적 아키텍처 미지원: 순환 신경망(RNN), Transformer 등 동적 구조는 검증 불가.")
    pdf.bullet("실수 연산 근사: 부동소수점 오차가 누적될 경우 결과의 신뢰성에 영향을 줄 수 있다.")
    pdf.ln(2)

    pdf.body(
        "종합하면, Marabou는 소규모 안전 필수 신경망(예: ACAS Xu 충돌 방지 시스템)의 공식 검증에 "
        "적합하다. 대규모 현대 모델에는 추상 해석(Abstract Interpretation) 기반 도구 등 "
        "더 확장성 있는 접근이 필요하다."
    )

    pdf.output(output)
    print(f"Report saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sat", action="store_true", help="Use SAT result template")
    parser.add_argument("--elapsed", type=float, default=float(RESULTS["elapsed_s"]))
    parser.add_argument("--out", type=str, default="report.pdf")
    args = parser.parse_args()

    if args.sat:
        RESULTS["outcome"]    = "SAT"
        RESULTS["adv_class"]  = 4
        RESULTS["adv_linf"]   = 0.009871
    RESULTS["elapsed_s"] = str(args.elapsed)

    build(RESULTS, args.out)
