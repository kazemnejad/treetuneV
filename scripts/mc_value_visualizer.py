import html
import json
import os
import random
import sys
from pathlib import Path
from typing import List

import gradio as gr
import markdown
import numpy as np
import pandas as pd
import torch
from l2m4m import LaTeX2MathMLExtension

from verl.utils.fs import copy_to_local
from verl.utils.tokenizer import hf_tokenizer

sys.path.insert(0, os.getcwd())
from verl.protocol import DataProto

css = """

.scrollable-container {
    display: flex;
    overflow-x: auto;
    scroll-snap-type: x mandatory;
    gap: 10px;
    padding-bottom: 10px;
    white-space: nowrap;
}

.scrollable-box {
    position: relative;
    flex: 0 0 40%;
    max-width: 500px;
    max-height: 500px;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 5px;
    scroll-snap-align: start;
    display: flex;
    font-size: 12px;
    /* Key properties for wrapping */
    white-space: pre-wrap;     /* Let text wrap onto new lines */
    overflow-wrap: break-word; /* Break up long words if needed */
    word-wrap: break-word;     /* Older, but still good fallback */
    overflow-y: auto;
}

.scrollable-container::-webkit-scrollbar {
    height: 8px;
}

.scrollable-container::-webkit-scrollbar-track {
    border-radius: 5px;
}

.scrollable-container::-webkit-scrollbar-thumb {
    border-radius: 5px;
}

.step-text {
    margin-bottom: 15px;
}

.step {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 10px;
    margin: -5px;
    position: relative;
}
.step-text {
    margin-bottom: 15px;
}

.step-text pre {
    white-space: pre-wrap;     /* Ensure pre tags also wrap */
    word-wrap: break-word;     /* Break long words in pre tags */
    overflow-wrap: break-word; /* Modern property for pre tags */
}


.scrollable-box pre {
    white-space: pre-wrap;     /* Wrap text within code/pre blocks */
    word-wrap: break-word;
}

/* Small rectangle for numbers */
.number-box {
    position: absolute;
    top: 6px;
    right: 6px;
    background-color: #f5f5f5; /* Light gray background */
    color: #000;
    font-size: 12px;
    padding: 2px 6px;
    border-radius: 3px;
    box-shadow: 0 0 3px rgba(0,0,0,0.2);
    transition: opacity 0.15s linear; /* Smooth transition for opacity */
    will-change: opacity;
    backface-visibility: hidden;
    transform: translateZ(0);
}

.number-box:hover {
    opacity: 0.2; /* Becomes semi-transparent on hover */
}

/* Small box for advantage */

.advantage-box {
    position: absolute;
    top: 6px;
    right: 6px;
    background-color: #f5f5f5; /* Light gray background */
    color: #000;
    font-size: 12px;
    padding: 2px 6px;
    border-radius: 3px;
    box-shadow: 0 0 3px rgba(0,0,0,0.2);
    transition: opacity 0.15s linear;
    will-change: opacity;
    backface-visibility: hidden;
    transform: translateZ(0);
}

.advantage-box:hover {
    opacity: 0.2;
}

.value-box {
    position: absolute;
    top: 6px;
    right: 72px;
    background-color: #f5f5f5; /* Light gray background */
    color: #000;
    font-size: 12px;
    padding: 2px 6px;
    border-radius: 3px;
    box-shadow: 0 0 3px rgba(0,0,0,0.2);
    transition: opacity 0.15s linear;
    will-change: opacity;
    backface-visibility: hidden;
    transform: translateZ(0);
}

.value-box:hover {
    opacity: 0.2;
}

.reward-box {
    position: absolute;
    top: 6px;
    right: 138px;
    background-color: #f5f5f5; /* Light gray background */
    color: #000;
    font-size: 12px;
    padding: 2px 6px;
    border-radius: 3px;
    box-shadow: 0 0 3px rgba(0,0,0,0.2);
    transition: opacity 0.15s linear;
    will-change: opacity;
    backface-visibility: hidden;
    transform: translateZ(0);
}

.reward-box:hover {
    opacity: 0.2;
}
"""

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""


def _pre_process_inputs(
    pad_token_id,
    prompt_token_ids: torch.Tensor,
) -> torch.Tensor:
    # remove the left padding in the prompt token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    return prompt_token_ids[non_pad_index:]


def get_split_visualization_html(text: str, indices: list[int]) -> str:
    # Add the end of the text as the last index if it's not already there
    if indices[-1] != len(text):
        indices.append(len(text))

    # Colors for highlighting and font style for code-like appearance
    colors = ["yellow", "lightgreen", "lightblue", "lightcoral", "lightgrey"]
    font_style = "white-space: pre-wrap;"  # Added white-space: pre-wrap to preserve whitespaces and newlines

    # Prepare the parts of text and the HTML format
    parts = []
    for i in range(len(indices) - 1):
        segment = text[indices[i] : indices[i + 1]]
        # segment = segment.replace('\n', '<br>')  # Replace newline characters with HTML line breaks
        # Properly encode HTML to avoid issues with MathJax
        segment = segment.replace("<", "&lt;").replace(">", "&gt;")
        color = colors[i % len(colors)]  # Cycle through colors
        parts.append(f"<span style='background-color: {color}; {font_style}'>{segment}</span>")

    # Join all parts into a single string
    highlighted_text = "".join(parts)

    # Use display_html to allow MathJax to process LaTeX within the HTML
    html_output = f"""
    <div>
        {highlighted_text}
    </div>
    <script type="text/javascript">
        MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
    </script>
    """

    return html_output


def get_response_html(response_text, view_options_list):
    res_md = f"{response_text}"

    extensions = []
    if "Render Math" in view_options_list:
        extensions.append(LaTeX2MathMLExtension())
    if "Render Markdown" in view_options_list:
        for _ in range(20):
            try:
                res_html = markdown.markdown(res_md, extensions=extensions)
                break
            except Exception as e:
                continue
    else:
        res_html = f"<pre>{res_md}</pre>"

    return res_html


def clean_segment(segment: str) -> str:
    # Escape HTML tags in the segment
    return html.escape(str(segment))


def escape_markdown_specials(text: str) -> str:
    """Escape Markdown special characters without touching HTML.

    This function adds a backslash before characters that have special meaning in
    Markdown so that the raw text is displayed as-is. It is intentionally
    conservative and operates character-by-character.
    """
    if text is None:
        return ""
    specials = set("\\`*_{}[]()#+-.!|>~")
    escaped_chars = []
    for ch in str(text):
        if ch in specials:
            escaped_chars.append("\\" + ch)
        else:
            escaped_chars.append(ch)
    return "".join(escaped_chars)


def extract_steps_from_response(response_ids, states, tokenizer):
    """Extract steps from response_ids using states as breaking points"""
    steps = []
    step_indices = []

    # States should be relative to response_ids

    for i in range(len(states)):
        end_ids = states[i]
        start_idx = states[i - 1] if i > 0 else 0
        step_token_ids = response_ids[start_idx:end_ids]

        # Convert to text
        step_text = tokenizer.decode(step_token_ids)
        steps.append(step_text)

        # Store the start index for visualization
        step_indices.append(len(step_text) + (step_indices[-1] if len(step_indices) > 0 else 0))

    step_text = tokenizer.decode(response_ids[states[-1] :])
    steps.append(step_text)
    step_indices.append(len(step_text) + (step_indices[-1] if len(step_indices) > 0 else 0))

    assert len(steps) == len(states) + 1

    return steps, step_indices


def create_step_html(traj, state_idx, state, step_text, view_options_list, tokenizer):
    """Create HTML for a single step using the new data format"""

    # Extract step information
    states = traj["states"]
    mc_value = None
    mc_value = traj["mc_values"].get(state, None) if "mc_values" in traj else None
    advantages = traj["advantages"]
    step_advantage = advantages[(states[state_idx - 1] if state_idx > 0 else 0) : state]
    if len(step_advantage) > 0:
        step_advantage = np.unique(step_advantage).tolist()
        step_advantage = step_advantage[0] if len(step_advantage) == 1 else None
    else:
        step_advantage = None

    reward = traj["token_level_rewards"][state - 1] if state > 0 else None

    # Get MC rollouts if available
    mc_rollout_texts = []
    mc_returns = []

    if traj.get("mc_extra_info", None) is not None:
        mc_extra_info = traj["mc_extra_info"]
        if state in mc_extra_info:
            state_info = mc_extra_info[state]
            if "__mc_returns" in state_info:
                mc_returns = state_info["__mc_returns"]

            if "__mc_rollouts" in state_info:
                mc_rollouts = state_info["__mc_rollouts"]

                for rollout in mc_rollouts:
                    mc_rollout_texts.append(tokenizer.decode(rollout))

    # Create color gradients for value and advantage boxes
    def lerp_color(start_color, end_color, t):
        # Ensure t is between 0 and 1
        t = max(0, min(1, t))
        r = int(start_color[0] + (end_color[0] - start_color[0]) * t)
        g = int(start_color[1] + (end_color[1] - start_color[1]) * t)
        b = int(start_color[2] + (end_color[2] - start_color[2]) * t)
        return f"rgb({r},{g},{b})"

    assert len(mc_rollout_texts) == len(mc_returns)

    boxes = []
    for mc_rollout_text, mc_return in zip(mc_rollout_texts, mc_returns):
        boxes.append(
            f"""
        <div class="scrollable-box">
            <div class="number-box"><strong style="color: {lerp_color((0, 0, 0), (0, 170, 0), mc_return)}">Return: {mc_return:.1f}</strong></div>            
            <pre style="font-family: inherit">{clean_segment(mc_rollout_text)}</pre>
        </div>
        """
        )
    boxes_html = "\n".join(boxes)
    if len(boxes_html) > 0:
        boxes_html = f"""
        <details>
            <summary style="font-size: 0.65em; color: #999999">Show MC Rollouts (k={len(mc_rollout_texts)})</summary>
            <div class="scrollable-container">
                {boxes_html}
            </div>
        </details>
        """

    # Value color (white to green)
    if mc_value is not None:
        value_t = max(0, min(1, mc_value))
        value_color = lerp_color((255, 255, 255), (0, 255, 0), value_t)
        value_html = f"<div class='value-box' style='background-color: {value_color}'>Val: {mc_value:.2f}</div>"
    else:
        value_html = ""

    # Advantage color (red to white to green)
    advantage_color = "white"
    advantage_text = "N/A"
    if step_advantage is not None:
        if not isinstance(step_advantage, list):
            advantage_text = f"{step_advantage:.2f}"
            if step_advantage < 0:
                adv_t = max(0, min(1, -step_advantage))
                advantage_color = lerp_color((255, 255, 255), (255, 0, 0), adv_t)
            else:
                adv_t = max(0, min(1, step_advantage))
                advantage_color = lerp_color((255, 255, 255), (0, 255, 0), adv_t)
        else:
            advantage_text = f"{step_advantage}"

    # Reward box
    reward_text = f"{reward:.2f}" if reward is not None else "N/A"
    reward_html = f"<div class='reward-box'>Rew: {reward_text}</div>"

    return f"""
        <div class="step">
            <div class="step-text">
                <div class="advantage-box" style="background-color: {advantage_color}">Adv: {advantage_text}</div>
                {value_html}
                {reward_html}
                <pre style="font-family: inherit; font-size: inherit">{clean_segment(step_text)}</pre>
            </div>

            {boxes_html}
        </div>
        """


def main():
    max_num_steps = 300
    state = {}
    step_components = []

    with gr.Blocks(title="TreeTune MC Value Visualizer", css=css, js=js_func) as demo:
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("# TreeTune MC Value Visualizer")
                run_path_input = gr.Textbox(
                    label="Enter rollout data directory",
                    placeholder="~/scratch/treetune_next/experiments/EXP-..../seed_2746318213",
                    scale=3,
                    lines=1,
                    autoscroll=False,
                    max_lines=1,
                    value="",
                )
                with gr.Row():
                    run_load_btn = gr.Button("Load", interactive=True, scale=1, variant="huggingface")
                    clear_btn = gr.Button("Clear", interactive=True, scale=1)
                with gr.Column() as run_info_column:
                    run_info = gr.Markdown(container=True, visible=False)
                    iteration_selector_dropdown = gr.Dropdown(
                        label="Select Iteration",
                        choices=[],
                        interactive=False,
                    )
                    iteration_select_btn = gr.Button("Select")

            with gr.Column(scale=5, show_progress=True) as example_column_parent:
                with gr.Column(visible=False) as example_column:
                    eval_title = gr.Markdown()
                    with gr.Row(equal_height=True):
                        example_id_slider = gr.Slider(minimum=0, maximum=99, step=1, label="Example ID", scale=8)
                        random_example_btn = gr.Button("Random\nExample", scale=2)
                    problem_html = gr.HTML()
                    view_options = gr.CheckboxGroup(
                        choices=["Render Markdown", "Render Math", "Visualize Split"],
                        value=["Render Markdown"],
                        label="View Options",
                        interactive=True,
                    )
                    for step_id in range(max_num_steps):
                        step_html = gr.HTML(
                            value="",
                            elem_id=f"step-{step_id}",
                            padding=False,
                            # visible=False,
                        )
                        step_components.append(step_html)

        def load_run(run_path: str, progress=gr.Progress()):
            if run_path == "":
                return gr.skip()

            progress(0, desc="Loading Run")

            run_path = str(run_path).replace("~", str(Path.home()))
            run_path = Path(run_path)

            if not run_path.is_dir():
                print(f"Run path {run_path} is not a directory")
                return gr.skip()

            state["run_path"] = run_path

            name = run_path.name
            info = f"### Name\n{name}"

            # Get the full_episodes directory path
            full_episodes_path = run_path / "full_episodes"

            iterations = []
            if full_episodes_path.exists() and full_episodes_path.is_dir():
                iterations = [d.name for d in full_episodes_path.iterdir() if d.is_dir() and d.name.startswith("iter_")]
                iterations.sort()

            progress(100)

            return {
                iteration_selector_dropdown: gr.Dropdown(choices=iterations, visible=True, interactive=True),
                run_info: gr.Markdown(info, visible=True),
                run_info_column: gr.Column(visible=True),
                example_column: gr.Column(visible=False),
            }

        def load_iteration_view(iteration_name: str, progress=gr.Progress()):
            run_path = state["run_path"]
            iteration_path = run_path / "full_episodes" / iteration_name

            progress(0, desc="Loading Evaluation")

            # Load the new format data
            batch_data = DataProto.load_from_disk(iteration_path / "batch.pkl")
            mc_extra_info_data = DataProto.load_from_disk(iteration_path / "mc_extra_info.pkl")

            # Store the data in state
            state["batch_data"] = batch_data
            state["mc_extra_info_data"] = mc_extra_info_data

            # Get tokenizer
            model_path = batch_data.meta_info["config"]["actor_rollout_ref"]["model"]["path"]
            use_shm = batch_data.meta_info["config"]["actor_rollout_ref"]["model"].get("use_shm", False)
            local_path = copy_to_local(model_path, use_shm=use_shm)
            state["tokenizer"] = hf_tokenizer(local_path)

            title = gr.Markdown(f"# Iteration: {iteration_name}", visible=True)

            progress(100)

            return {
                eval_title: title,
                example_id_slider: gr.Slider(
                    minimum=0, maximum=len(batch_data) - 1, value=-1, step=1, label="Trajectory ID"
                ),
                example_column: gr.Column(visible=True),
                iteration_selector_dropdown: gr.skip(),
            }

        def update_example(traj_ix: int, view_options_list):
            state["example_id"] = traj_ix

            batch_data = state["batch_data"]
            mc_extra_info_data = state["mc_extra_info_data"]
            tokenizer = state["tokenizer"]

            # Get trajectory data from the new format
            traj = batch_data[traj_ix]
            traj_mc_extra = mc_extra_info_data[traj_ix] if traj_ix < len(mc_extra_info_data) else None
            if traj_mc_extra is not None:
                traj_mc_extra = traj_mc_extra.non_tensor_batch["mc_extra_info"]

            # Extract data from the new format
            prompt_ids = _pre_process_inputs(tokenizer.pad_token_id, traj.batch["prompts"]).cpu().numpy()

            # Extract response_ids using attention_mask
            # Find where attention_mask is 1 (actual tokens)
            response_mask = traj.batch["response_mask"]
            response_length = response_mask.sum(-1)
            response_ids = traj.batch["responses"][:response_length].cpu().numpy()
            response_text = tokenizer.decode(response_ids)

            token_level_rewards = traj.batch["token_level_rewards"].cpu().numpy()

            # Get states from non_tensor_batch
            states = traj.non_tensor_batch["states"].tolist()
            if states[0] != 0:
                states = [0] + states

            mc_values = traj.non_tensor_batch["mc_values"]
            advantages = traj.batch["advantages"].numpy()

            # Other info
            data_source = traj.non_tensor_batch["data_source"]
            ability = traj.non_tensor_batch["ability"]

            traj_reward = token_level_rewards[response_length - 1]

            try:
                ground_truth = traj.non_tensor_batch["reward_model"]["ground_truth"]
            except:
                ground_truth = None

            try:
                extra_info = traj.non_tensor_batch["extra_info"]
                answer_key = [k for k in ["answer", "solution"] if k in extra_info][0]
                answer = extra_info[answer_key]
            except:
                answer = None

            # Extract steps from response_ids using states
            steps, step_indices = extract_steps_from_response(response_ids, states, tokenizer)

            # Prepare trajectory data for visualization
            traj_data = {
                "prompt_ids": prompt_ids,
                "data_source": data_source,
                "ability": ability,
                "ground_truth": ground_truth,
                "response_ids": response_ids,
                "states": states,
                "mc_values": mc_values,
                "advantages": advantages,
                "steps": steps,
                "step_indices": step_indices,
                "mc_extra_info": traj_mc_extra,
                "token_level_rewards": token_level_rewards,
            }

            # Create problem description
            pr_md = ""
            pr_md += f"## Problem (Trajectory ID: {traj_ix})\n"

            # Try to get problem text from meta_info or other sources
            prompt_text = tokenizer.decode(prompt_ids)
            pr_md += prompt_text.replace("\n", "<br>") + "\n"

            if ground_truth is not None:
                pr_md += f"## Ground Truth\n{ground_truth}\n"
                if answer is not None:
                    pr_md += f"\n\n<details>\n<summary>Answer</summary>\n<p>{answer}</p>\n</details>\n"

            # Get initial state value
            initial_state = states[0] if len(states) > 0 else 0
            value_at_initial_state = mc_values.get(initial_state, "N/A")
            if value_at_initial_state != "N/A":
                value_at_initial_state = f"{value_at_initial_state:.2f}"

            stats_info = [
                f"Outcome Reward: {traj_reward:.2f}",
                f"Number of Steps: {len(steps)}",
                f"Number of States: {len(states)}",
            ]
            if data_source is not None:
                stats_info.append(f"Data Source: {data_source}")
            if ability is not None:
                stats_info.append(f"Ability: {ability}")

            pr_gen_stats = " | ".join(stats_info)
            if len(pr_gen_stats) > 0:
                pr_md += f"## Info\n{pr_gen_stats}"

            visualize_split = "Visualize Split" in view_options_list

            if not visualize_split:
                pr_md += f"""
                <details>
                    <summary>Response</summary>
                    {html.escape(str(response_text))}
                </details>
                """

            extensions = []
            if "Render Math" in view_options_list:
                extensions.append(LaTeX2MathMLExtension())
            if "Render Markdown" in view_options_list:
                pr_html = markdown.markdown(pr_md, extensions=extensions)
            else:
                pr_html = f"<pre>{pr_md}</pre>"

            if visualize_split:
                response_text = tokenizer.decode(response_ids)
                split_html = get_split_visualization_html(response_text, step_indices)
                split_html = f"""
                <details>
                    <summary>Response</summary>
                    {split_html}
                </details>
                """
                pr_html += split_html

            steps_updates = {comp: gr.update(visible=False) for comp in step_components}
            for state_idx in range(len(states)):
                steps_updates[step_components[state_idx]] = gr.HTML(
                    create_step_html(
                        traj_data, state_idx, states[state_idx], steps[state_idx], view_options_list, tokenizer
                    ),
                    visible=True,
                )

            # Final Step
            steps_updates[step_components[len(states)]] = gr.HTML(
                create_step_html(traj_data, len(states), len(response_ids), steps[-1], view_options_list, tokenizer),
                visible=True,
            )

            return {
                problem_html: gr.HTML(pr_html, visible=True),
                **steps_updates,
            }

        run_load_btn.click(
            load_run,
            inputs=[run_path_input],
            outputs=[
                run_info_column,
                example_column_parent,
                example_column,
                iteration_selector_dropdown,
                run_info,
            ],
        )

        iteration_select_btn.click(
            load_iteration_view,
            inputs=[iteration_selector_dropdown],
            outputs=[eval_title, example_id_slider, example_column, iteration_selector_dropdown],
        ).then(lambda: gr.Slider(value=0), outputs=[example_id_slider])

        example_id_slider.change(
            update_example,
            inputs=[example_id_slider, view_options],
            outputs=[problem_html, *step_components],
        )

        random_example_btn.click(
            lambda x: gr.Slider(value=random.randint(0, len(state["batch_data"]) - 1)),
            inputs=[random_example_btn],
            outputs=[example_id_slider],
        )

        view_options.change(
            update_example,
            inputs=[example_id_slider, view_options],
            outputs=[problem_html, *step_components],
        )

        clear_btn.click(
            lambda: {
                example_column: gr.update(visible=False),
                iteration_selector_dropdown: None,
                run_info: gr.update(visible=False),
            },
            outputs=[
                run_info_column,
                example_column_parent,
                example_column,
                iteration_selector_dropdown,
                run_info,
            ],
        )

    demo.launch()


if __name__ == "__main__":
    main()
