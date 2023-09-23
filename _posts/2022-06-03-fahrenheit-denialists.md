---
layout: distill
title: "&#x1f964; Lofi Beats to Scale and Rotate to"
img: /assets/img/fahrenheit_denialists/screenshot_scene_1.png
date: 2022-06-03
tags: graphics, physics_engine
categories: project

authors:
    - name: Jordan Lin
      url: "https://kuanhenglin.github.io"
      affiliations:
        name: UCLA CS
    - name: Joice He
      url: "https://www.linkedin.com/in/joice-he/"
      affiliations:
        name: UCLA Ling & CS
    - name: Aidan Cini
      url: "https://www.linkedin.com/in/aidan-cini/"
      affiliations:
        name: UCLA CS

toc:
  - name: Features
    subsections:
        - name: Toggle Pause
        - name: Toggle Bounding Boxes
        - name: Toggle Blender
        - name: Shoot Object
        - name: Angular Velocity and Impulse
        - name: Initialize Scene
        - name: Ungroup objects
        - name: Gravity
        - name: Movement
  - name: Running the Code

---

> We created a 3D sandbox scene of the "lofi beats to relax/study to" girl with a fully-fledged physics engine, rigged models, and shadowing in `tiny-graphics.js`. This is our final project for UCLA's CS 174A: Introduction to Computer Graphics, Spring 2022, taught by [Dr. Asish Law](https://www.linkedin.com/in/asishlaw/).

<div class="repo p-2 text-center github-repo-in-post">
  <a href="https://github.com/kuanhenglin/fahrenheit-denialists" rel="external nofollow noopener" target="_blank">
    <img class="repo-img-light w-100" alt="kuanhenglin/fahrenheit-denialists" src="https://github-readme-stats.vercel.app/api/pin/?username=kuanhenglin&amp;repo=fahrenheit-denialists&amp;theme=default&amp;show_owner=true">
    <img class="repo-img-dark w-100" alt="kuanhenglin/fahrenheit-denialists" src="https://github-readme-stats.vercel.app/api/pin/?username=kuanhenglin&amp;repo=fahrenheit-denialists&amp;theme=dark&amp;show_owner=true">
  </a>
</div>

<img src="/assets/img/fahrenheit_denialists/screenshot_scene_1.png" width="100%" />

Our project is a 3D sandbox scene of the widely beloved study girl from the 24/7 Youtube radio "[lofi hip hop radio - beats to relax/study to](https://youtu.be/5qap5aO4i9A)." When you start up the scene, you will see the study girl sitting in her chair just vibing to the chill beats. On her desk is a laptop, a notebook, and a study lamp. To her right is a window looking out to the night sky with a purring cat on the windowsill. We hope that our scene with give you the motivation to get though *finals*! Everything seems calm and peaceful until you unpause and unleash the underlying Physics engine that Jordan created.

**Jordan Lin:** Physics engine and shadowing <br/>
**Joice He:** Modelling, art, texturing <br/>
**Aidan Cini:** Scene setup

## Features

The following is a list of features of our little demo.

### Toggle Pause

Initially the scene is static, meaning that time is completely stopped. Entering `Ctrl + p` will cause time to resume. You should see that the study girl and the table start vibrating, almost like they are vibing to the dope beats. This occurs because small impulses in the Physics engine can quickly snowball into larger impulses, which is a common problem with resting objects in Physics engines that do not dampen tiny motions&mdash;our engine!

### Toggle Bounding Boxes

<img src="/assets/img/fahrenheit_denialists/screenshot_bounding_1.png" width="100%" />

Entering `Ctrl + b` turns on the bounding boxes of the objects, allowing the user to see how the objects collide. All bounding boxes are oriented bounding boxes ( rectangular prisms of arbitrary size and 3D rotation.) It is the most fun to watch the bounding boxes as the objects go flying around after unpausing.

These bounding boxes act as the collision/hit boxes in our Physics engine.

### Toggle Blender

<img src="/assets/img/fahrenheit_denialists/screenshot_blender_1.png" width="100%" />

Entering `Ctrp + d` spawns a massive rectangular prism that spins around the floor and pushes everything in the scene around in the most chaotic way&mdash;we call this a blender. The blender is not for the faint of heart; if you care about the study girl’s safety then please **DO NOT USE**.

### Shoot Object

Entering `Ctrl + e` spawns ellipsoidal objects into the scene at roughly random positions and velocities (but all generally pointed towards the girl). When paused, this is not too exciting as the objects just stay static in the scene, but unpause and you can watch all the new objects fly around the scene.

### Angular Velocity and Impulse

<img src="/assets/img/fahrenheit_denialists/screenshot_angular_1.png" width="100%" />

When angular velocity and impulse is enabled with `Ctrl + 6`, the Physics engine enables angular collision resolution and collisions now affect the rotation and angular velocity of objects. Now, this is not particularly stable, because gravity currently only works on the center-of-mass of objects/object groups instead of all parts of the object. Something something snowballing, and chaos ensues.

When angular velocity and impulse is disabled with `Ctrl + 7`, the Physics engine disables angular collision resolution. This is the *default* of our demo and is a bit more stable, though still not terribly stable. Stay paused if you want peace, though that is not at all the objective of this demo.

### Initialize Scene

Entering `Ctrl + i` reinitializes the scene to its original state, which is useful for clearing objects. If the scene and Physics engine crashes (mostly due to spawning too many objects), the initialize scene button will not work, and you will need to refresh the webpage.

### Ungroup objects

Without Hitboxes  |  With Hitboxes
:---:|:---:
<img src="/assets/img/fahrenheit_denialists/screenshot_ungroup_1.png" width="100%" />  |  <img src="/assets/img/fahrenheit_denialists/screenshot_ungroup_2.png" width="100%" />

Do you want to see the study girl break into pieces? If so, this is the feature for you. Since the study girl is a very complicated object to model with complex collision boxes, we modeled her body by breaking it into many different simpler shapes. This button ungroups the individual components of grouped objects (e.g., study girl, desk, chair, etc.) and allows them to move and collide freely.

### Gravity

<img src="/assets/img/fahrenheit_denialists/screenshot_gravity_1.png" width="100%" />

You can change gravity to any one of the coordinate axes. Simply click the button that displays the direction you want gravity to go and it will follow. `none` causes the objects to float about, `+x` makes the objects fly backward, `-x` pushes the objects toward the brick wall, `+y` makes the objects fly to the ceiling, `-y` is normal gravity pushing objects to the floor, `+z` makes the objects come toward the camera, and `-z` makes the objects fly towards the window wall.

### Movement

<img src="/assets/img/fahrenheit_denialists/screenshot_scene_2.png" width="100%" />

As per a feature which comes with `tiny-graphics.js`, you can move around the scene with `w`, `a`, `s` and `d`; you can also move up with `Space` and move down with `z`.

## Running the Code

Clone this [repository](https://github.com/kuanhenglin/fahrenheit-denialists) to your local machine.

For Windows: run/double-click the `host.bat` file in the main directory. <br/>
For MacOS: run the `host.command` file. Alternatively, execute `python3 server.py` or `python3 server.py` (should work for both Windows and MacOS). <br/>
If you are using Linux, you can probably figure this out yourself.

Then, type `localhost:8000` into your browser, hit enter and have fun :D

We use [`tiny-graphics.js`](https://github.com/encyclopedia-of-code/tiny-graphics-js.git), which is like [`three.js`](https://threejs.org/) but worse.

*Have fun exploring our demo!*