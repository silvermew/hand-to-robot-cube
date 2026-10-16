# From my hand to a robot arm: turning a cube

Following the directives given by Humanoid where I needed to train a world model and a VLA using self filmed videos as a guiding policy this is what I have done:

I filmed my own hand picking up a Rubik's cube, turning it, and putting it back down, and later pushing it between
three crosses on the table. I tracked the hand and the
cube in 3D using ArUco markers, replayed the motion on a simulated Panda arm, and measured where the robot does something different from
my hand. Then I trained a small world model on my data, used it to plan, and fine-tuned a vision-language-action
model (SmolVLA) to do the task from camera images, eventually from 19 different sentences: turn the cube, carry it
to another cross, or push it there.

Everything robot-side happens in simulation (robosuite). The data is 109 short videos of my hand from a stereo camera
pair, and 23 phone videos of pushes.

![human and sim side by side](outputs/real/gifs/best_clip_62_rot+60_A.gif)

## Results summary

| | Result |
|---|---|
| Clips recorded / usable for replay | 109 / 109 |
| Sim replay, gripper turns like my hand | 63/109 (58%) succeed |
| Sim replay, gripper turns like the cube | 96/109 (88%) succeed |
| How much more the cube turns than my hand | 1.16x counter-clockwise, 1.51x clockwise |
| World model, best learned model vs simple gain | 12.2 vs 9.2 deg error (the gain wins) |
| Planning with the world model vs naive plan | 65% vs 100% |
| SmolVLA, one instruction, 50 new cube poses | 84% (two runs: 42/50 and 42/50) |
| SmolVLA, three instructions, same 50 poses | 76% / 40% / 88% |
| Push moves filmed with a phone | 42 (28 straight, 14 turning) |
| Sim replay of my pushes, straight and turning | 27/28 and 0/14 succeed |
| SmolVLA, 19 sentences, 20 poses each (v11) | pushes 46%, pick and place 23% |

A replay or policy episode counts as a success when the cube is lifted at least 2 cm, put back within 3 cm, and
turned within 15 degrees of the target. A push counts when the cube ends within 3 cm of where my push put it,
turned within 15 degrees of the target, without being lifted or tipped.

## How the pieces fit

```mermaid
flowchart LR
    subgraph hand [My hand]
        V1[109 stereo clips] --> T1[tracking<br/>track.py]
        T1 --> X1[grasp and release<br/>extract.py]
        V2[23 phone videos] --> T2[one-camera tracking<br/>push_track.py]
        T2 --> X2[pushes and moves<br/>push_extract.py]
    end
    subgraph robot [Simulated Panda]
        R1[grasp replay<br/>sim_replay.py]
        R2[push replay<br/>sim_push.py]
    end
    subgraph learn [Learning]
        WM[world model<br/>world_model/] --> PL[planning<br/>planning.py]
        DM[demos<br/>make_sim_demos.py] --> DS[LeRobot dataset<br/>to_lerobot.py]
        DS --> VLA[SmolVLA fine-tune<br/>scripts/vla/]
        VLA --> EV[closed-loop test<br/>eval_vla.py]
    end
    X1 --> R1
    X1 --> WM
    PL --> R1
    X2 --> R2
    R1 --> DM
    R2 --> DM
```

The robot never learns from my videos directly. My tracked motion drives a scripted controller in simulation, the
replays that succeed become demonstrations, and SmolVLA learns from those. The world model learns from my data
only.

## Setup

- **Cameras:** I used a calibrated stereo camera pair that was used in a previous project, I faced the cameras and used a black table to reduce image noise. They record at
  30 fps. Using two cameras helped me reduce the error on depth for the videos.
- **Markers:** ArUco markers on the table (the world frame), on a carton piece stuck on the back of my hand, and on five faces
  of a classic Rubik's cube.
- **Recordings:**
  - 75 clips where I turn the cube on the spot by -90 to +120 degrees, on three marked spots;
  - 18 clips where I carry it to another spot;
  - 16 clips where I tried to do the movement faster to limit test the robot;
  - later, 23 phone videos of pushes (section 9).

## How it works, and what went wrong along the way

### 1. Tracking (`src/track.py`, `src/stereo.py`)

**How it works:**
- Every frame, both cameras detect the markers.
- The cube's pose is the one that best explains the marker corners seen by both cameras at once. The hand's pose
  is found the same way.
- One camera alone is fine for the yaw angle (0.2 degree error) but not for position, so positions only come from
  frames that both cameras see.

**Problems I hit:**
- **The top marker was glued 180 degrees off my convention.** Every switch between markers flipped the cube's
  angle. I found it with a 3D check from the two cameras.
- **One camera picked the mirror solution of the table marker.** A flat marker has two possible poses, and one
  camera sometimes picks the wrong one, which put the resting cube 14 cm above the table. The table pose now comes
  from both cameras.
- **A thin white margin on the black table can make a marker read too big.** In a test render it read 8% too
  big: the edge of the margin forms a second, larger square that also decodes as the marker. The detector now
  checks that the inside of each square is dark.
- **My hand often covered the cube while holding it.** With my palm on top, no camera can see the cube. Holding
  it by the sides helped (tracked through the hold in 60% of clips instead of 48%), but in about half of the clips
  the cube is hidden while held.

### 2. When does the hand grasp and release? (`src/extract.py`)

**How it works:**
- The cube's resting pose before my hand appears and after it leaves gives two anchors.
- The grasp is the last moment the cube is still at rest; the release is the first moment it rests again.
- When the cube is hidden at those moments, the hand decides: while it holds the cube on the table it sits at a
  steady "holding height", and the grasp and release are when it leaves and returns to that height.

**Problems I hit:**
- My first rule ("the hand's lowest point") failed on fast turns, where there is no pause.
- It also failed on carries, where the hand is at a different height at the other spot.

The final rule finds both times in all 109 clips. I checked it by eye on height plots in `outputs/real/phases/`.

### 3. Replay on a simulated robot (`src/sim_replay.py`, `src/replay_all.py`)

**How it works:**
- The robot is a scripted controller driven by my data, not a learned policy.
- Between grasp and release, the gripper follows the point of my hand that sat at the cube's centre when I
  grasped it, and turns with my hand.
- The approach and the retreat are scripted (straight down from 10 cm, straight up).

**Problems I hit:**
- **The first real replay succeeded on 0 of 27 clips.** My hand arrives from the side at table height, after
  resting flat on the table. A gripper copying that sweeps along the table and pushes the cube away. Scripting the
  approach fixed it.
- **The robot still under-turned the cube in most clockwise clips.** The sim cube turns exactly as much as my
  hand, but my cube turned more, because my fingers roll it. A parallel gripper cannot do that.
  - This is the embodiment gap, and it depends on the direction: the cube turns 1.16 times my hand's turn
    counter-clockwise and 1.51 times clockwise.
  - Replays succeed 33/48 counter-clockwise and only 8/36 clockwise.

**Fix:** turning the gripper by the cube's measured turn (known from where the cube started and ended) instead of
my hand's raises the replay to 96/109. Clockwise goes from 8/36 to 30/36. What still fails is the robot's own
limits: fast clips where my hand was lost mid-turn, two +120 degree turns, and a few placements 3-8 cm off, mostly on
carries.

![embodiment gap](outputs/real/embodiment_gap.png)

### 4. World model (`src/world_model/`)

**What it is:** a small model that predicts the cube's angle 0.1 s ahead from the cube's angle, my hand's angle,
my hand's change in angle, and whether the cube is held. No images.

**Compared:**
- an MLP and a GRU, five of each;
- three baselines: the cube does not move; the cube turns like the hand; the cube turns a fixed multiple of the
  hand's turn.

**Data:** 64 clips where the cube is visible while held, split by clip into 49 train, 7 validation and 8 test.

| Test, error in degrees | 1 s ahead | 3 s ahead | whole hold |
|---|---|---|---|
| cube does not move | 23.8 | 72.9 | 67.4 |
| cube turns like the hand | 5.0 | 14.1 | 13.5 |
| cube turns 1.14 x hand | 2.9 | 7.7 | **9.2** |
| MLP x5 | 3.4 | 13.6 | 13.7 |
| GRU x5 | 3.4 | 12.1 | 12.2 |

**What this shows:**
- **A single number beats both learned models.** The cube mostly turns with my hand, scaled up by my fingers.
- **The GRU wins only when extrapolating:** trained on turns up to 90 degrees and tested on +120 degree turns, it
  gets 9.3 degrees against the gain's 10.2.
- **The five copies' disagreement does flag bad predictions:** rank correlation 0.33. The fifth of predictions
  where they disagree most average 6-7 degrees of error, against 2 degrees for the rest.

### 5. Planning with the world model (`src/planning.py`)

**How it works:** for a target turn, the planner picks the hand turn that the world model predicts will give it,
and replays that in simulation.

**Result:** it succeeds 65% of the time. Simply turning the robot by the target succeeds 100%.

**Why:** the world model learned how the cube moves in my hand, where the fingers add turn, so it plans smaller
turns. The robot's gripper turns the cube exactly as much as its wrist. Planning through a model of the human is the
embodiment gap again, seen from the model's side.

### 6. SmolVLA from camera images (`src/make_sim_demos.py`, `src/to_lerobot.py`, `src/eval_vla.py`, `scripts/vla/`)

**The demos:** I replayed my +60 degree turns in simulation from many random cube positions and angles. Each
successful replay became a demonstration: two camera images, the robot state, and the robot's actions.

**The model:** I fine-tuned SmolVLA (`lerobot/smolvla_base`) on these demos.
- Only its 100M-parameter action part learns; the vision-language part stays frozen.
- Training ran on one 8 GB RTX 2080 at about 1.2 s per step, in full precision (mixed precision crashed with this
  model).

**The test:** 50 cube poses the demos never contained. The scripted controller succeeds on all 50.

| Version | What changed in the demos | Success on the 50 test poses |
|---|---|---|
| v1 | 100 demos from 10 of my clips | 4% |
| v3 | 300 demos from one clip, plus the time since the start in the robot state | 60% |
| v4 | 600 demos | 62% and 68% (two runs) |
| v5 | plus random grasp face near the "switch" angles (below) | 52% and 56% |
| v6 | 900 new demos: higher approach (15 cm) with a 0.4 s pause above the cube | **84% and 84%** |

**What I learned at each step:**
- **v1 hovered above the cube, turned it the wrong amount, or pushed it.**
  - 10 different human clips gave 10 different right answers for the same first image, so the model blended them.
  - The scripted demos also pause at fixed times, which a single image cannot tell apart from moving.
  - Using one clip and giving the model the clock fixed most of this (v3).
- **Some ideas made it worse.**
  - Planning more often (every 10 or 25 steps instead of 50) dropped success to 0-5%.
  - Adding random drift to the demos, so they would show corrections (v2, not in the table), gave 0%: the
    model learned to wander.
- **The renders showed the grasp failing near four cube angles** (below). The script picks which pair of faces
  to grab from the cube's angle, and that choice flips by 90 degrees at four angles. Next to a flip, the model
  blended two grasps 90 degrees apart and closed on the corners. Within 10 degrees of a flip, v4 succeeded 2-4
  times out of 9; elsewhere 66-78% of the time.
- **I also noticed in the videos that the arm sometimes came down too early and pushed the cube.** A higher
  approach with a short pause above the cube (v6) cut failed grasps from 11-14 per 50 to 2.
  - v5's random face choice alone did not help.
  - v6 changed both at once, so I cannot fully separate the two.

![grasp failure](docs/vla_grasp_failure.jpg)

*Top row: a success, where the fingers are square to the cube in the wrist view (right of each pair). The other
rows: the gripper arrives about 45 degrees off, closes on the corners, and the cube tips.*

![v6 test episode](docs/vla_v6_episode.gif)

One caveat: v6 was chosen among three versions using the same 50 test poses, so 84% is slightly optimistic. The jump
from v4 is far beyond the noise of about 7%.

### 7. Three instructions, one policy (v7)

**The idea:** a labmate's model follows sentences like "put the cube on the cylinder". I have no human demos of
stacking, and scripting that would teach the robot my code instead of my hand. So I used three sentences that each
match motions I filmed, one clip per sentence, 300 demos each:
- counter-clockwise: my +55 degree turn;
- clockwise: a -57 degree turn (the gripper follows the cube's turn, as in section 3);
- no turn: a +4 degree "turn".

The model was fine-tuned from v6 and tested with its last checkpoint on the same 50 poses.

| Sentence | Success | Lifted | Turned right | Median turn (demos) |
|---|---|---|---|---|
| "pick up the cube, rotate it counter-clockwise and put it down" | 76% | 94% | 78% | +37 (+48) |
| "pick up the cube, rotate it clockwise and put it down" | 40% | 98% | 42% | -41 (-57) |
| "pick up the cube and put it back down without turning it" | 88% | 94% | 96% | +6 (+5) |

**What works:** the robot follows the sentence. From the same images it turns one way, the other way, or not at
all, and whenever it lifted the cube it never turned the wrong way.

**What does not:** it under-turns, which is why the clockwise sentence scores only 40%. Sharing one policy also cost
the counter-clockwise task some (76% against 84% alone). This run got only half a pass over its data, so more
training is the obvious next step.

![same start, three sentences](docs/vla_three_instructions.gif)

### 8. Carrying the cube between crosses

**How it works:**
- I drew the three crosses of my table (A, B, C) on the sim table. My table is turned onto the sim table by one
  fixed rotation, so a carry from A to C in my video goes from A to C in the sim. Before, each replay was turned to
  fit the sim cube's angle, which loses the direction.
- One of my clips per sentence, 13 sentences: turn on the spot counter-clockwise or clockwise, carry between any
  two crosses (6), and carry while turning (5). 100 demos each, the cube on its start cross within 2 cm, at any angle.
- Carries copy the cube's turn, not my hand's: on some plain carries my hand drifted about 20 degrees.

**Problem I hit:** one carry, C to A with a -96 degree turn, never replayed. I moved fast, the scripted arm trailed
my hand by 8 cm and put the cube down 5 cm off, so I left that sentence out.

These 1300 demos were never trained on alone: the push demos were ready first, so they went into v9 (section 10).

### 9. Pushing, filmed with a phone (`src/push_track.py`, `src/push_extract.py`, `src/sim_push.py`)

**Recordings:** the robot's cameras were down that day, so I filmed with my phone on a stand, from about where the
robot head stands. The 23 videos contain 42 moves:
- 20 straight pushes between neighbouring crosses (5 per direction) and 8 long ones between A and C;
- 14 turning moves (8 clockwise, 6 counter-clockwise), each made of 2 to 4 separate pushes that turn the cube
  about 90 degrees each.

**How the tracking works with one camera:**
- One camera gets a marker's depth 2-3 cm wrong. But a pushed cube stays on the table, so its top marker lies in the
  plane 5.7 cm above the table marker: each corner is where the camera ray through it meets that plane.
- I had no checkerboard video of the phone, so its focal length is the one that puts the top marker, from its own
  one-camera pose, at the cube's height. The positions barely depend on it.
- Where the cube rests on a cross, it is 0.1-1.7 cm from that cross as the stereo cameras measured it the week
  before (up to 3.5 cm where I stopped short).
- The videos are cut into pushes where the cube moves, and into moves from one cross to the next.

**Replay on the simulated robot:** a scripted controller again, but driven by the cube's motion this time: my
fingertips are not tracked (the marker on my hand sits 12 cm behind them). The gripper is closed, its fingers sit on
the face away from the motion at mid-height, and they follow the path that point took on my cube.

**Problems I hit:**
- **The sim arm sagged 3.4 cm toward its base** from a still target near the table, enough to push the cube 3 cm
  off-centre and turn it. A small integral term on the target fixed it.
- **Every push tipped the cube by 14 degrees.** The controller holds the gripper about 6 degrees off vertical, so the
  fingers touch the top of the face first, and at robosuite's friction (1.0) a 5.7 cm cube tips when pushed above
  2.85 cm. The push scenes use 0.4 (plastic on cloth, not measured). The finger pads keep their own friction, so
  grasps do not change.
- **Fingertips lower than 1.6 cm above the table wedged under the cube** and held it tilted.

**Result:** 27 of my 28 straight pushes replay, median 1.4 cm off; the one failure is a long push where the sim cube
spun 73 degrees. **None of my 14 turning moves replays.**
- Each of my turning pushes is close to a turn about one point near a front corner: the cube "walks" forward about
  5 cm per 90 degrees.
- The sim cube turned only 41% (clockwise) to 54% (counter-clockwise) as much, and tipped over in 4 moves.
  Changing the contact point, the push angle and height, or pushing with a corner of the fingers never turned it
  more than about 50 degrees in one push.
- I think there are two reasons. MuJoCo rests a box on its four corners, which resists turning the most, and the
  closed Panda fingers are a flat 4.6 cm block that keeps the face square like a ruler, where my fingertip is a
  point the cube can pivot around. This is the embodiment gap again, in the contact this time instead of the grasp.

### 10. Nineteen sentences, one policy (v9 to v11)

**The data:** the 13 sentences of section 8 plus 6 new ones, "push the cube from cross X to cross Y", each taught by
one of my straight pushes, 100 demos each. "Move" and "push" use the same crosses, so only the verb decides between
picking the cube up and sliding it.

**The test:** 20 cube poses per sentence, the cube on its start cross. With 20 poses a rate is good to about 10%.

| Version | What changed | Pushes (6 sentences) | Pick and place (13) |
|---|---|---|---|
| v9 | 30k steps from v7 on all 19 sentences | 18% | 26% |
| v10 | push demos pause above the start point and come down 4.6 cm from the face instead of 2.6 | 15% | 22% |
| v11 | push demos start coming down only after the policy's first 2.5 s of actions | **46%** | 23% |

**What I learned:**
- **v9 knocked the cube over** in most episodes of four push sentences, all the pushes starting at A or C: the
  closed gripper came down on the cube's top edge and rolled it.
- **v10's pause did not help, and a test showed why.** The policy predicts 50 steps (2.5 s) at a time and runs them
  blind. In the push demos the descent started at step 39, so the first chunk, predicted from the starting pose,
  already brought the fingers down: in a test they landed 0.5 cm from the face instead of 4.6.
- **v11 starts the descent after 2.6 s,** as the grasp demos do after 2.4 s, so it is predicted from right above the
  start point with the wrist camera on the cube. No push episode tipped the cube after that (0 of 120), and the four
  failing sentences went from 0% to 30-60%. It is the lesson of v6, learned a second time.
- **The verb works.** With "push" the cube stayed on the table in all 120 push episodes; with "move" between the
  same crosses the robot lifted it in 50-85% of episodes.
- **What does not work yet:** pushes still end 4-10 cm off when the cube turns on the way, and most carries stop
  3-7 cm short of the far cross while the scripted demos land within 1 cm. One, A to C while turning, puts the cube
  back down where it started. So 6 of the 11 carry sentences score 0-5%.
- The two turn-on-the-spot sentences dropped from v7's 76% and 40% to 30% and 35%. The test is not the same (the
  cube now starts on a cross), and the model now shares itself with 17 other sentences.

![same start, two verbs](docs/vla_move_vs_push.gif)

*Same start cross, two sentences (v11): "move" picks the cube up and carries it, "push" slides it.*

<details>
<summary>v11, every sentence</summary>

| Sentence | Success | Lifted (pick) or kept low (push) | Scripted replay, same poses |
|---|---|---|---|
| push the cube from cross A to cross B | 60% | 100% | 90% |
| push the cube from cross B to cross C | 45% | 100% | 95% |
| push the cube from cross C to cross B | 50% | 100% | 100% |
| push the cube from cross B to cross A | 45% | 100% | 100% |
| push the cube from cross A to cross C | 45% | 100% | 75% |
| push the cube from cross C to cross A | 30% | 100% | 60% |
| pick up the cube, rotate it counter-clockwise and put it down | 30% | 50% | 100% |
| pick up the cube, rotate it clockwise and put it down | 35% | 85% | 100% |
| move the cube from cross A to cross B | 50% | 85% | 100% |
| move the cube from cross B to cross C | 5% | 55% | 100% |
| move the cube from cross A to cross C | 0% | 60% | 100% |
| move the cube from cross C to cross A | 0% | 75% | 100% |
| move the cube from cross B to cross A | 0% | 75% | 100% |
| move the cube from cross C to cross B | 30% | 80% | 100% |
| move the cube from cross A to cross B and rotate it counter-clockwise | 45% | 75% | 100% |
| move the cube from cross B to cross C and rotate it clockwise | 30% | 85% | 100% |
| move the cube from cross A to cross C and rotate it counter-clockwise | 0% | 50% | 100% |
| move the cube from cross B to cross A and rotate it counter-clockwise | 70% | 75% | 100% |
| move the cube from cross C to cross B and rotate it clockwise | 0% | 80% | 100% |

</details>

## Limitations

- **One person, one cube, one table,** and only turns about the vertical axis.
- **The push videos come from one phone camera** with an estimated focal length, not a calibrated one, and the
  push friction (0.4) is a guess. Turning pushes do not work in the simulator at all.
- **About half the clips have the cube hidden while held,** so the world model uses 64 of 109.
- **The grasp and release rules are hand-tuned thresholds.**
- **The robot side is simulation only.**
- **The VLA learns from one template motion per sentence,** with the clock in its state, and is tested on 20-50
  simulated poses per sentence.

## How to run

The environments and their versions are in `SETUP.md`.

```bash
python -m src.run_all                  # raw videos -> tracking, QC, grasp/release, replays, world model, planning (~25 min)
python -m pytest tests -q              # geometry and convention tests
python -m src.run_all --synthetic      # the same pipeline on rendered test clips with known ground truth
python -m src.push_track               # phone videos -> cube and hand tracks in the table frame
python -m src.push_extract             # tracks -> pushes and moves (outputs/push/moves.csv)
python -m src.sim_push --all           # replay every move on the simulated robot (outputs/push/replay.csv)
```

**VLA:** the runs are in `scripts/vla/` (`run_v3.sh` to `run_v7.sh`). Each one generates demos
(`src.make_sim_demos`), converts them to a LeRobot dataset (`src.to_lerobot`), trains (`train_smolvla.sh`) and
evaluates (`src.eval_vla`). For the 19 sentences, `run_v8_data.sh` makes the pick-and-place demos, and each of
v9 to v11 has a data script (push demos, merged with v8's by `src.merge_datasets`) and a training script that also
runs the tests. Evaluation results are in `outputs/vla/`.

**Reproducibility:** I checked that `run_all` rebuilds every number above from the raw videos. The raw videos,
demos, datasets and model checkpoints are not in the repository because of their size.

## Repository layout

```
src/            pipeline: markers, tracking, grasp/release, replay, pushing, world model, planning, VLA data and evaluation
scripts/vla/    the SmolVLA runs (demos, conversion, training, evaluation)
tests/          unit tests, synthetic clip generator, simulator smoke tests
outputs/real/   results on my clips: replay tables, embodiment-gap plots, world model report and plots
outputs/push/   my push moves and their sim replays
outputs/vla/    SmolVLA evaluation results per version
outputs/print/  printable marker sheets
docs/           images and GIFs used here
shot_list.csv   the list of clips I recorded
```
