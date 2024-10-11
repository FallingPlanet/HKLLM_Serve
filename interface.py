# interface.py
import gradio as gr
import sqlite3

DATABASE_FILE = "processed_data.db"

def fetch_next_sample(mode):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, text, classification FROM processed_samples
        WHERE user_feedback IS NULL AND mode=?
        LIMIT 1
    ''', (mode,))
    row = cursor.fetchone()
    conn.close()
    if row:
        sample_id, text, classification = row
        return sample_id, text, classification
    else:
        return None, "No more samples.", ""

def update_sample_feedback(sample_id, user_feedback, user_classification=None):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE processed_samples
        SET user_feedback=?, user_classification=?
        WHERE id=?
    ''', (user_feedback, user_classification, sample_id))
    conn.commit()
    conn.close()
    
def submit_feedback_dpo(feedback_value):
    if feedback_value == "Reject Both":
        user_classification.visible = True
    else:
        chosen_option = "Option 1" if feedback_value == "Accept Option 1" else "Option 2"
        update_sample_feedback(sample_id.value, feedback_value, chosen_option)
        load_sample()

def interface():
    with gr.Blocks() as demo:
        with gr.Tab("KTO Mode"):
            sample_id = gr.Variable()
            text = gr.Textbox(label="Text", interactive=False)
            classification = gr.Textbox(label="Model Classification", interactive=False)
            feedback = gr.Radio(["Accept", "Reject"], label="Feedback")
            user_classification = gr.Textbox(label="Your Classification", visible=False)
            submit_button = gr.Button("Submit Feedback")

            def load_sample():
                id_, text_value, classification_value = fetch_next_sample("KTO")
                sample_id.value = id_
                text.value = text_value
                classification.value = classification_value
                user_classification.visible = False

            def submit_feedback(feedback_value):
                if feedback_value == "Reject":
                    user_classification.visible = True
                else:
                    update_sample_feedback(sample_id.value, feedback_value)
                    load_sample()

            def submit_user_classification(classification_value):
                update_sample_feedback(sample_id.value, "Reject", classification_value)
                load_sample()

            submit_button.click(submit_feedback, inputs=feedback, outputs=None)
            user_classification.submit(submit_user_classification, inputs=user_classification, outputs=None)
            load_sample()

        with gr.Tab("DPO Mode"):
            # Similar implementation with DPO specifics
            pass

        with gr.Tab("Auto Mode"):
            # Display processed data
            pass

        with gr.Tab("Settings"):
            # Allow changing model settings, bit precision, LoRA adapters, number of instances
            pass

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    interface()
