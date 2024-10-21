import base64
import os

from fastapi import FastAPI

from optimus1.server.agent import AgentFactory
from optimus1.server.api.request import MCRequest, MCResponse
from optimus1.server.api.utils import base64_to_image, base64lst2img_path

IMAGE_ROOT = "imgs"

app = FastAPI()
agent = AgentFactory.get_agent(
    plan_with_gpt=True,
    plan_model=None,
    in_model="checkpoints/vpt/2x.model",
    in_weights="checkpoints/steve1/steve1.weights",
    prior_weights="checkpoints/steve1/steve1_prior.pt",
)


def _img2base64(img_path: str):
    with open(img_path, "rb") as f:
        img = base64.b64encode(f.read())
    return img.decode("utf-8")


def _filter_task_obs(task: str) -> str:
    """
    Filter the task observations based on the given task.

    Args:
        task (str): The task to filter the observations for.

    Returns:
        str: The path of the first image that matches the given task.

    """
    task = task.replace(" ", "_")
    task_imgs = [img for img in os.listdir(IMAGE_ROOT) if ".jpg" in img and task in img]
    task_imgs.sort(key=lambda x: int(x.split("_")[-1].replace(".jpg", "")))
    return os.path.join(IMAGE_ROOT, task_imgs[0])


@app.get("/reset")
def reset() -> MCResponse:
    global agent
    agent = AgentFactory.reset()
    print("agent reset")
    return MCResponse(response="reset done")


@app.post("/chat")
def chat(req: MCRequest) -> MCResponse:
    global agent

    if req.type is None:
        req.type = "plan"
    rgb_obs = base64_to_image(
        req.rgb_images,
        image_root=IMAGE_ROOT,
        task=req.task_or_instruction,
        step=req.current_step,
    )

    match req.type:
        case "retrieval":
            retry = 0
            while True:
                try:
                    plans_retrieval = agent.retrieve(
                        req.task_or_instruction,
                        rgb_obs[-1],
                    )
                    response = MCResponse(response=plans_retrieval)
                    break
                except:
                    retry += 1
                    print("connection error, retry: ", retry)
        case "plan":
            retry = 0
            while True:
                try:
                    plans = agent.plan(
                        req.task_or_instruction,
                        rgb_obs[-1],
                        req.example,
                        req.visual_info,
                        req.graph,
                    )
                    response = MCResponse(response=plans)
                    break
                except:
                    retry += 1
                    print("connection error, retry: ", retry)
        case "action":
            import time

            start = time.perf_counter()
            minrl_action = agent.action(req.task_or_instruction, rgb_obs)
            end = time.perf_counter()
            print(end - start, " s")  # 0.04s
            response = MCResponse(response=minrl_action)
            print(response)
        case "reflection":
            old_obs = _filter_task_obs(req.task_or_instruction)
            print(f"old_obs {old_obs} current step {req.current_step}")
            retry = 0

            done_imgs, cont_imgs, replan_imgs = (
                req.done_imgs,
                req.cont_imgs,
                req.replan_imgs,
            )
            done, cont, replan = (
                base64lst2img_path(done_imgs),
                base64lst2img_path(cont_imgs),
                base64lst2img_path(replan_imgs),
            )
            while retry < 10:
                try:
                    reflection = agent.reflection(
                        req.task_or_instruction, old_obs, rgb_obs[-1]
                    )
                    print(f"{old_obs} <-> {rgb_obs[-1]}: {reflection}")
                    response = MCResponse(
                        response=reflection, appendix=_img2base64(old_obs)
                    )
                    break
                except:
                    retry += 1
                    time.sleep(1)
                    print("connection error, retry: ", retry)
        case "replan":
            retry = 0
            while retry < 10:
                try:
                    replan = agent.replan(
                        req.task_or_instruction,
                        rgb_obs[-1],
                        req.error_info,
                        req.example,
                        req.graph,
                    )
                    response = MCResponse(response=replan)
                    print(replan)
                    break
                except Exception as e:
                    retry += 1
                    time.sleep(1)
                    print(f"connection error {e}, retry: {retry}")
        case _:
            response = MCResponse(message=f"{req.type} not support...", status_code=400)
    return response
