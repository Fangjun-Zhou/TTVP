import gradio as gr
import title_predict_api
import cv_api
import image_scaler
from PIL import Image as II
from si_prefix import si_format


def title_predict(Title):
    return si_format(title_predict_api.title_predict_view(Title), precision=2)


def image_predict(Image):
    return si_format(
        cv_api.predict_image(
            image_scaler.scale_image(
                Image
            )
        ), precision=2
    )


text = gr.Interface(
    fn=title_predict,
    inputs=gr.Textbox(lines=2, placeholder="Enter your title here"),
    outputs="text",
    allow_flagging=False
)

image = gr.Interface(
    fn=image_predict,
    inputs=gr.Image(),
    outputs="text",
    allow_flagging=False,
    server_port=80
)

demo = gr.TabbedInterface(
    [text, image], ["Title2View", "Image2View"])
demo.launch(share=True, server_port=7860, server_name="0.0.0.0")
