"""Gradio frontend for the Titanic Survival Classifier."""

from pathlib import Path
import gradio as gr
import joblib
import pandas as pd

from train import FEATURE_COLUMNS, MODELS_DIR, train_and_save_models

MODEL_PATHS = {
    "Decision Tree": MODELS_DIR / "decision_tree.joblib",
    "Binary Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
}


def load_models() -> dict[str, object]:
    """Load persisted models, training them first if missing."""
    if not all(path.exists() for path in MODEL_PATHS.values()):
        train_and_save_models()
    return {name: joblib.load(path) for name, path in MODEL_PATHS.items()}


MODELS = load_models()


def predict_survival(
    model_name: str,
    pclass: int,
    sex: int,
    age: float,
    sibsp: int,
    parch: int,
    fare: float,
    embarked: int,
) -> tuple[dict[str, float], str]:
    """Predict Titanic passenger survival probability and summary text."""
    input_df = pd.DataFrame(
        [
            {
                "Pclass": int(pclass),
                "Sex": int(sex),
                "Age": float(age),
                "SibSp": int(sibsp),
                "Parch": int(parch),
                "Fare": float(fare),
                "Embarked": int(embarked),
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    model = MODELS[model_name]
    probs = model.predict_proba(input_df)[0]
    pred = int(model.predict(input_df)[0])

    label_probs = {
        "Selamat": float(probs[1]),
        "Tidak Selamat": float(probs[0]),
    }
    status = "Selamat" if pred == 1 else "Tidak Selamat"
    summary = (
        f"**Model:** {model_name}  \n"
        f"**Prediksi:** {status}  \n"
        f"**Peluang Selamat:** {probs[1] * 100:.1f}%"
    )
    return label_probs, summary


def build_demo() -> gr.Blocks:
    """Create the Gradio Blocks interface."""
    with gr.Blocks(title="Titanic Survival Classifier") as demo:
        gr.Markdown(
            "# Titanic Survival Classifier\n"
            "Prediksi keselamatan penumpang Titanic menggunakan model "
            "**Decision Tree** atau **Binary Logistic Regression**."
        )

        with gr.Row():
            with gr.Column():
                model_name = gr.Radio(
                    choices=["Decision Tree", "Binary Logistic Regression"],
                    value="Decision Tree",
                    label="Pilih Model",
                )
                pclass = gr.Radio(
                    choices=[
                        ("1 - First Class", 1),
                        ("2 - Second Class", 2),
                        ("3 - Third Class", 3),
                    ],
                    value=3,
                    label="Kelas Tiket (Pclass)",
                )
                sex = gr.Radio(
                    choices=[("Laki-laki", 1), ("Perempuan", 0)],
                    value=1,
                    label="Jenis Kelamin (Sex)",
                )
                age = gr.Slider(
                    minimum=0,
                    maximum=80,
                    step=1,
                    value=25,
                    label="Usia (Age)",
                )
                with gr.Row():
                    sibsp = gr.Number(
                        value=0,
                        minimum=0,
                        maximum=10,
                        precision=0,
                        label="Saudara / Pasangan (SibSp)",
                    )
                    parch = gr.Number(
                        value=0,
                        minimum=0,
                        maximum=10,
                        precision=0,
                        label="Orang Tua / Anak (Parch)",
                    )
                with gr.Row():
                    fare = gr.Number(
                        value=15.0,
                        minimum=0.0,
                        maximum=600.0,
                        label="Harga Tiket (Fare)",
                    )
                    embarked = gr.Dropdown(
                        choices=[
                            ("Cherbourg (C)", 1),
                            ("Queenstown (Q)", 2),
                            ("Southampton (S)", 3),
                            ("Tidak Diketahui", 0),
                        ],
                        value=3,
                        label="Pelabuhan Keberangkatan (Embarked)",
                    )
                submit_btn = gr.Button("Prediksi", variant="primary")

            with gr.Column():
                label_output = gr.Label(label="Probabilitas Keselamatan")
                summary_output = gr.Markdown(label="Ringkasan")

        gr.Examples(
            examples=[
                ["Decision Tree", 1, 0, 38, 1, 0, 71.28, 1],
                ["Decision Tree", 3, 1, 22, 1, 0, 7.25, 3],
                ["Binary Logistic Regression", 2, 0, 14, 1, 0, 30.07, 1],
            ],
            inputs=[model_name, pclass, sex, age, sibsp, parch, fare, embarked],
        )

        submit_btn.click(
            fn=predict_survival,
            inputs=[model_name, pclass, sex, age, sibsp, parch, fare, embarked],
            outputs=[label_output, summary_output],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
