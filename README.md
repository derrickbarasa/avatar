# avatar

A stylised 3D avatar creator. The character is generated in code from smooth
meshes and drawn with a cel-shading shader and inked outlines. Pick every
feature from a tabbed side panel, strike a pose, and export the result.

- **Face:** a skull with a flat-sided cranium, sloping forehead, jaw line and chin (not a ball), with nose, cheekbones, eyes with a procedural iris,
  eyelids and lashes that blink, ears, nostrils, lips with teeth and tongue, six
  expressions, freckles, glasses, earrings.
- **Hair:** fourteen styles built from thousands of real strands (grown from the
  scalp, clumped into locks, pushed out of the head, shoulders and chest, with
  anisotropic highlights): short, curly, long, bob, bangs, bun, ponytail, quiff,
  mohawk, afro, pigtails, braid, undercut, wavy. Facial hair is a full beard (sideburns, cheeks, chin and under the jaw, thinning toward
  the cheek line, clear of the lips), or short stubble or a goatee), a moustache, and eyebrows, all strands hugging the skin.
  Plus hats (beanie, cap, bucket, top hat, beret, headband): the hair under a hat is only what hangs below its edge.
- **Body:** three body types, build, height, shoulder and hip width, head size and neck
  length. The torso has a sloping collar, shoulder blades, chest, waist and buttocks. Arms and legs have elliptical, tapering sections (deltoid, biceps, elbow, forearm flattening
  to the wrist, thigh, knee, calf bulging backwards, ankle) with a slight bend, hands with a
  tapering palm, rounded knuckles and fingertips and a thumb that curl into a fist, and shaped feet that join the ankle.
- **Outfit:** tee, polo, long sleeve, sweater, turtleneck, tank, hoodie, jacket, dress; jeans, leggings, wide
  trousers, shorts, skirt;
  sneakers, boots or barefoot; stripes, dots, plaid or a chest emblem.
- **Extras:** necklace, scarf, watch, backpack (a rounded bag with a lid flap and pocket, straps over the shoulders).
- **Poses and life:** relaxed, A-pose, wave, hands on hips, cheer, walk and dance,
  plus idle animation: breathing sway, head movement, blinking and darting
  eye glances.
- **Talking:** type a line (or pick a quick phrase) and the avatar says it with
  your system voice. The lips form real mouth shapes (open, spread, pucker,
  pressed, teeth and tongue) timed to the audio, and while it talks it nods on
  stressed syllables, raises its brows, gestures with its hands, holds eye
  contact and turns to face you.
- **Live:** the avatar can follow your face from a webcam (head, eyes, blinks, mouth, brows),
  follow your microphone (mouth, nods and gestures), and appear as a virtual webcam in Zoom, Teams, Meet
  or OBS.
- **Presets and sharing:** eight outfit presets, and a short share code that
  captures a whole avatar.

## Run

```bash
pip install -r requirements.txt
python avatar.py
```

Or `pip install .` to get an `avatar-studio` command. Optional
(`requirements-optional.txt`): OpenCV 4 for face detection in the photo import,
`imageio-ffmpeg` for MP4 clips, and trimesh / scikit-image for extra tests.
`python build_exe.py` makes a standalone app with PyInstaller (`dist/AvatarStudio/`, about
300 MB; built and launched on Windows only).

## Saving and your library

Everything you make lives in one folder, `~/Documents/AvatarStudio` (override with
`AVATARKIT_HOME`), with `avatars/`, `exports/` and `packs/` inside.

- **Autosave:** your work is restored the next time you start (`last.avatar`).
- **Library tab:** name the avatar, **Save** / **Save as**, and reopen or delete
  from the gallery of thumbnails. **Open** picks any `.avatar` file; you can also
  drag one (or a share code text file) onto the window.
- **Undo / redo** (Ctrl+Z / Ctrl+Y) for every change.
- **Share code:** a short `AV2-...` string that captures the whole avatar (C copies
  it, Ctrl+V loads one).
- `.avatar` files are plain JSON with a thumbnail, so they are easy to back up.

## Controls

| Input | Action |
| --- | --- |
| Click | pick a swatch or chip; click a tab; drag the scene to orbit |
| Tab / Shift+Tab | switch panel tab |
| Arrows | move the focus between options / change the focused option |
| Wheel | scroll the panel, or zoom when over the scene |
| Ctrl+S / Ctrl+O | save to the library / open a file |
| Ctrl+Z / Ctrl+Y | undo / redo |
| H | show or hide the shortcut sheet |
| T / Enter | open the Talk tab and type / speak the text (again to stop) |
| Esc | stop speaking (then quit) |
| R | randomize everything |
| [ / ] | previous / next outfit preset |
| V | cycle camera: bust / face / hands / full body |
| Space | toggle auto-spin |
| S / G | save PNG / transparent PNG |
| E / O / B | export GLB / OBJ + MTL / full bundle (zip) |
| C / Ctrl+V | copy share code / load from the clipboard |
| M / W / F | microphone lip-sync / virtual webcam / face tracking on or off |
| P | pick a photo and set skin tone, hair colour and length, facial hair |
| K / L | save / load `avatar.json` |

## Interface

A floating card holds eight tabs (Face, Hair, Body, Outfit, Extras, Scene, Talk,
Library). Colour options are swatches, everything else is a row of chips; long tabs
scroll. Short toasts confirm saves and exports. The window can be resized.

## Command line

```bash
python avatar.py --shot me.png --bare --view face --set hair=Long --set skin=Tan
python avatar.py --shot pose.png --view full --set pose=Wave --time 0.3
python avatar.py --shot logo.png --transparent --preset Streetwear
python avatar.py --sheet 12 --shot gallery.png --seed 42     # contact sheet of random avatars
python avatar.py --code AV2-101...                           # start from a share code
python avatar.py --photo selfie.jpg                          # suggest options from a photo
python avatar.py --say "Hello, I can talk now."              # start speaking straight away
python avatar.py --shot mouth.png --say "Hello there" --time 0.4   # the lips 0.4 s into the line
python avatar.py --shot ooh.png --view face --viseme OO      # hold one mouth shape
```

## Live: face tracking, microphone and virtual webcam

Switch them on in the **Talk** tab (Live section), with **F** / **M** / **W**, or from the command line
(`--track`, `--mic`, `--webcam`). They need optional packages: `pip install mediapipe` for face tracking,
`sounddevice` for the microphone and `pyvirtualcam` for the virtual webcam.

- **Face tracking:** the avatar copies your head turn, nod and tilt, your blinks, eye direction, mouth
  (open, wide, smile) and brow height. It opens the camera and a face-landmark model (MediaPipe's face
  mesh) in the background, then spends about a second learning your neutral face, so look at the camera
  with a relaxed face when it starts. It works as a mirror: turn your head to your right and the avatar
  turns toward the right of the screen (`--no-mirror` for the other way). `--camera N` picks the camera,
  `--list-cameras` lists them. While you speak a line or use the microphone, those drive the mouth instead.

- **Microphone:** the mouth follows what you say, with no text needed. Loudness sets how far it opens,
  the vowel (the sound's two main frequency peaks) picks "ah", "oo" or "ee" shapes, and hissy sounds give
  a thin spread mouth. The level adapts to your microphone and room. The head nods and the hands gesture
  while you speak, and it idles when you stop. `--mic-device NAME` (or a number) chooses the input;
  `--list-mics` shows them.
- **Virtual webcam:** sends a 16:9 head-and-shoulders picture (`--webcam-size`, default 1280x720,
  `--webcam-fps`, default 30) to a virtual camera. On Windows and macOS install OBS Studio and start its
  Virtual Camera once, then pick "OBS Virtual Camera" in your meeting app. On Linux load `v4l2loopback`.
  Keep the Avatar Studio window open (not minimised) while the camera is in use.
  The Live chip shows the frame rate it is really achieving. The scene is drawn a second time for the
  camera, so on a slower graphics card choose **Fast** quality in the Scene tab or a smaller size
  (`--webcam-size 640x360`) to get a smoother picture.

Use both together to be the avatar on a call: your voice drives the mouth.

## Talking

The **Talk** tab has the text box, Speak / Random line buttons, quick phrases,
a voice picker, speech speed and a gestures switch. Speech uses what the
operating system already has, so there is nothing to install: Windows SAPI
voices (via PowerShell), `say` on macOS, or `espeak-ng` / `espeak` on Linux.
Synthesis runs in the background, then the audio is analysed: the text is turned
into mouth shapes, stretched over the part of the audio where someone is
speaking, and scaled by loudness so the mouth closes in pauses and opens wider on
stressed syllables. If no voice or audio device is available the avatar still
mimes the sentence, timed from the text alone.

`--set OPTION=VALUE` takes the option keys and value names shown in the panel
(for example `hair=Mohawk`, `pattern=Stripes`, `pose=Hands on hips`).
`python avatar.py --help` lists every flag.

## Exports

All exports go to `exports/` in the library folder (Library tab: *Show last export*).
Long hair and beards are thinned on export to keep files a sensible size.

- **GLB** keeps the skeleton as glTF nodes so you can repose it in Blender, Unity or Godot.
- **Skinned GLB** has a real armature (bones, inverse bind matrices, joints and weights).
- **Animated GLB** is the skinned model plus animations: the current pose's loop (walk and
  dance are full cycles, other poses idle for four seconds) and the nine emotes, each as its own
  clip. The bundle's skinned GLB carries the same animations.
- **VRM 1.0** is a humanoid with the standard bone map, in metres, feet on the ground, plus the
  expression presets as morph targets: happy, angry, sad, surprised, relaxed, the mouth shapes
  aa / ih / ou / ee / oh, and blink / blinkLeft / blinkRight. They are measured from your avatar's
  own face, so it starts from the expression you picked.
- **OBJ + MTL** bakes the current pose into one static mesh per part.
- **Bundle** is a zip with every format, the `.avatar` file and a preview.
- **PNG / transparent PNG** is the viewport.
- **Clip (MP4 or GIF):** records the avatar speaking the current line, with the audio
  track. MP4 needs ffmpeg (system-installed or `pip install imageio-ffmpeg`); without it
  you get a GIF.

The GLB/skinned/VRM files pass the Khronos glTF validator with no errors. Materials are
plain colours; procedural extras (iris, freckles, cloth patterns) are render-only.

## Content packs

Drop a `.json` file into `packs/` to add hair, eye, cloth, pants and shoe colours,
outfit presets, quick phrases and backgrounds. See `examples/packs/neon.json`. Broken
packs are reported at startup and never stop the good ones loading.

## More features

- Nine emotes (wave, laugh, shrug...), emotion picked from the text being spoken.
- Lip-sync rules for English, Spanish, Italian and Portuguese (other languages use the
  English rules with accents removed).
- Soft shadows with ambient occlusion (under the chin, hair and arms), cloth folds, and hair
  that swings when the head moves (Shadows / Hair physics in the Scene tab).
- PNG exports and `--bare` screenshots are drawn 3x larger off-screen and shrunk for smooth
  edges; PNG exports are twice the window's pixel size.
- **Quality** (Scene tab): High, Balanced or Fast. Lower settings draw fewer hair strands, use a
  smaller shadow filter and drop ambient occlusion, for slower graphics cards. Exported pictures
  always use the best quality. `python tests/bench_gui.py` prints frame times for each setting.
  Meshes are uploaded to the graphics card once and drawn from there, which made drawing about 2x faster.

## Not done yet

Neural/offline TTS, BVH import, VRM look-at, landmark-based
photo matching, tracking more than one face, and true screen-space ambient occlusion (the current one is a
soft overhead shadow). Automated tests run on Windows, macOS and Linux CI, but the interactive
window has only been used on Windows.

## Photo import

`P` (or `--photo`) reads skin tone, hair colour, hair length and facial hair
from a picture. It is a colour heuristic, not face recognition: with OpenCV
installed it locates the face first, otherwise it assumes a centred face. It
works best on an evenly lit, front-facing portrait. It corrects overall exposure
but not coloured lighting, and it won't detect eye colour, glasses or face shape.

## Layout

```text
avatar.py            entry point
avatarkit/
  options.py         option tables, presets, share codes, save/load
  mesh.py mathutil.py   mesh primitives (tube, loft, ellipsoid...) and math
  head.py hair.py    head, face details, hair layers, hats, facial hair
  strands.py         strand growth (physics-lite) and strand mesh builder
  body.py            torso, limbs, clothes, hands, accessories (as rig nodes)
  rig.py poses.py    skeleton, poses, idle animation, gaze and talking gestures
  builder.py         assembles an avatar from options
  render.py          GLSL cel shader, outline pass, camera, capture
  ui.py app.py       interface overlay (card, chips, toasts), input, command line
  exporters.py       OBJ, GLB, skinned GLB and VRM writers
  store.py history.py   library folder, .avatar files, autosave, undo/redo
  packs.py           content packs
  physics.py         hair swing
  recorder.py        MP4 / GIF clip recording
  photo.py           photo -> option suggestions
  speech.py tts.py   text -> mouth shapes and lip-sync timeline; system text-to-speech
  live.py            microphone lip-sync and the virtual webcam
  tracking.py        face tracking: landmarks -> head, eyes, mouth and brows
tests/               python -m unittest discover -s tests
```
