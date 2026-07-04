import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mujoco
import imageio


XML = """
<mujoco>
  <option timestep="0.02" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3"/>
    <camera name="cam" pos="1.2 -1.2 0.9" xyaxes="1 1 0 -0.4 0.4 1"/>
    <geom type="box" pos="0 0 0" size="0.6 0.4 0.03" rgba="0.6 0.6 0.6 1"/>

    <body name="cube" pos="-0.25 0 0.08">
      <joint name="cube_free" type="free"/>
      <geom type="box" size="0.04 0.04 0.04" rgba="0.1 0.4 1 1"/>
    </body>

    <body name="gripper" mocap="true" pos="-0.25 0 0.25">
      <geom type="sphere" size="0.035" rgba="1 0.2 0.2 1"/>
    </body>

    <geom type="cylinder" pos="0.25 0 0.04" size="0.06 0.005" rgba="0.2 1 0.2 1"/>
  </worldbody>
</mujoco>
"""


HORIZON = 8
EXECUTE_STEPS = 4


def get_state(gripper, cube, target):
    return np.concatenate([gripper, cube, target]).astype(np.float32)


def set_cube_pos(data, pos):
    data.qpos[:3] = pos
    data.qpos[3:7] = np.array([1, 0, 0, 0])


def expert_action(gripper, cube, target):
    above_cube = cube + np.array([0, 0, 0.18])
    grasp_pos = cube + np.array([0, 0, 0.06])
    above_target = target + np.array([0, 0, 0.18])
    place_pos = target + np.array([0, 0, 0.08])

    holding = cube[2] > 0.12
    close_xy = np.linalg.norm(gripper[:2] - cube[:2]) < 0.03

    if not holding:
        goal = grasp_pos if close_xy else above_cube
    else:
        goal = place_pos if np.linalg.norm(gripper[:2] - target[:2]) < 0.03 else above_target

    return np.clip(goal - gripper, -0.03, 0.03).astype(np.float32)


def step_toy_world(gripper, cube, action, target, holding):
    gripper = gripper + action

    if np.linalg.norm(gripper - (cube + np.array([0, 0, 0.06]))) < 0.06:
        holding = True

    if holding:
        cube = gripper - np.array([0, 0, 0.06])

    if holding and np.linalg.norm(cube[:2] - target[:2]) < 0.04 and cube[2] < 0.12:
        holding = False

    return gripper, cube, holding


class ACTPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, HORIZON * 3)
        )

    def forward(self, state):
        action_chunk = self.net(state)
        return action_chunk.reshape(-1, HORIZON, 3)


# ------------------------------------------------------------
# 1. Generate expert action chunks
# ------------------------------------------------------------

target = np.array([0.25, 0.0, 0.08])
all_states = []
all_chunks = []

for episode in range(100):
    cube = np.array([
        np.random.uniform(-0.35, -0.15),
        np.random.uniform(-0.12, 0.12),
        0.08
    ])

    gripper = cube + np.array([0, 0, 0.22])
    holding = False

    states = []
    actions = []

    for t in range(140):
        state = get_state(gripper, cube, target)
        action = expert_action(gripper, cube, target)

        states.append(state)
        actions.append(action)

        gripper, cube, holding = step_toy_world(
            gripper, cube, action, target, holding
        )

    states = np.array(states)
    actions = np.array(actions)

    for i in range(len(states) - HORIZON):
        all_states.append(states[i])
        all_chunks.append(actions[i:i + HORIZON])

all_states = torch.tensor(np.array(all_states), dtype=torch.float32)
all_chunks = torch.tensor(np.array(all_chunks), dtype=torch.float32)


# ------------------------------------------------------------
# 2. Train ACT with supervised learning
# ------------------------------------------------------------

policy = ACTPolicy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(300):
    pred_chunks = policy(all_states)
    loss = loss_fn(pred_chunks, all_chunks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"epoch {epoch}, loss = {loss.item():.5f}")


# ------------------------------------------------------------
# 3. Roll out ACT policy
# ------------------------------------------------------------

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

frames = []

cube = np.array([-0.28, 0.08, 0.08])
gripper = cube + np.array([0, 0, 0.22])
holding = False

for t in range(0, 160, EXECUTE_STEPS):
    state = torch.tensor(get_state(gripper, cube, target)).unsqueeze(0)

    with torch.no_grad():
        action_chunk = policy(state).squeeze(0).numpy()

    for action in action_chunk[:EXECUTE_STEPS]:
        action = np.clip(action, -0.03, 0.03)

        gripper, cube, holding = step_toy_world(
            gripper, cube, action, target, holding
        )

        data.mocap_pos[0] = gripper
        set_cube_pos(data, cube)
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera="cam")
        frames.append(renderer.render())

imageio.mimsave("act_pickplace.mp4", frames, fps=30)
print("Saved act_pickplace.mp4")