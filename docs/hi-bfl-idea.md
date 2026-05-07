# readme
Hi Black Forest Labs :) Glad you are here!

As I really wanna work with you I thought to myself to go on a little sidequest, eventough my first babysteps on the surrogate training in this repository are just done, and see how the flow matching approach could work with my Laser Powder Bed Fusion Simulation. As FM is one of the things that made you famous (and it is elegant af!), I wanna give it a go and see how it can be transfered.

For you to be able to follow my journey propperly, here is a little recap on what is done allready:
1. I implemented a Finite-Differences Method solver with PyTorch. Why? Differential Physics <3. I started with a simple moving gaussian heat source and stepwise improved with material lookup tabels (LUTs) and phase change maps. 
2. After that was done I tried to enhance the performance with Triton and this brought improvements of ~10.6x in speed and ~xxx in memory consumption. I only work on my local machine, a laptop with an 16GB vRAM eGPU running on windows (won't buy one of those again..) so a little speed and memory efficiency comes in handy. I decided on using WSL, as Triton requires Linux (well, it seems there are some trys to port it to Windows, but everything i read about it seemed like pure pain).
3. Based on this simulation I trained my first surrogate model, a basic Unet with... As my SSD is, lets say, tight, I descided on an experience buffer (after making the mistake to go with online learning, resp. simulating a step with my solver after each training step, this had multiple disadvantages besides a very slow training). Meaning I simulate some 80 steps and store it in my RAM. During training I randomly grab those samples and train my model. The results were okay'ish for the baseline.

Now! I found the guts, based on external influences, to apply to this awesome vacancy. And then I thought: "Well, I have a solver, I walked my first surrogate steps, I understand the Conditional Flow Matching method, lets marry my surrogate ambitions with generative powers.". Why this could be a match? The beauty of FM lies in the deterministic coupling. By guiding the vector field with the previous state as an conditional anchor, we maintain physical consistency without sacrificing the generative flexibility of the FM approach. In other words: We don't just jump between plausible snapshots, but follow a physically consistent trajectory.

## TL;DR

## Let the journey begin!
### A baseline
I started with implementing a first baseline version using flow matching with simplicity over complexity in mind, consisting of
1. *the VelocityNet (3D U-Net Backbone with 32 Base Channels)*: Chosen as lean starter architecture to balance spatial expressiveness with vRAM constraints of my local WSL2/eGPU setup. 
2. *SiLU Activation & AdaGroupNorm for smoothness and efficient conditioning*: SiLU ensures a continous and differenziable velocity field, while AdaGroupNorm allows the model to modulate its features based on external process parameters
3. A 128-Dim Conditioning space for the process and material parameters (including laser power, scan speed, and basic material properties like density and reference conductivity).

I did a smoke test with just a few data points from an old simulation, and it seemed to run through without exploding — providing a solid "functional skeleton" to build upon. This wasn't meant to be the final high-performance model, but a verification that the end-to-end pipeline (Data $\to$ Flow Matching $\to$ Stepper) is technically sound.

### The Dataset Horrors; or a Fan Speed Crime
Now I need data! Hell yeah, I do have a working solver, lets write a script randomizing the material LUTs, process parameters and hatch patterns (those describe how the lasers are moving in the plane) and get ~2000 samples, holistically sampled from the simulations with a moderatly high variance to begin with. With full confidence I startet my script! Aaand there is my datase... WHY IS MY GPU KEEPING CRASHING?

Yeah, why is my GPU crashing? And not just a simple crash, no easy diagnosis like OOM, it got geniounly disconnected from the laptop, there was a full blown driver hick up. In the notebook everything run through. So what's that? Now I learned something about WSL. Windows monitors GPU activity via the Timeout Detection and Recovery (TDR) watchdog. If a single GPU kernel occupies the device for more than ~2 seconds, Windows forcibly resets the driver. High-diffusivity materials (e.g., Ti-6Al-4V) trigger many more CFL sub-steps per macro-step, making individual `step_adaptive` calls hit this limit during long exposure times (≥50 µs). So my first attempt on solving this issue was chunking the stepping. Instead of calling `stepper.step_adaptive` once with the full `exposure_time`, the script iterates in chunks of at most **5 µs** each. 

I started the dataset generation again, but my stepper chunking didn't seem to be sufficient enough. I learned, that PyTorch kernel launches are asynchronous — `step_adaptive` returns to Python the moment kernels are *queued*, not when they *finish*. Without a sync point, the while loop enqueues the next chunk's kernels before the GPU has finished the previous chunk. Windows TDR measures time from when a GPU packet is submitted until the GPU acknowledges completion. Without sync, consecutive chunks appear as continuous uninterrupted GPU work and TDR fires. The synchronize call gives Windows the "breathing point" it needs to reset its TDR timer between slices.

But now, this must be it :) Nope... Now I'm fed up! Now I will force the TDR watch dog to wait longer, full 60 seconds, via the Windows registry before it sticks it's nose into something that's none of it's business. AGAIN A BIG NOPE!

And then I remembered, that I'm an engineer (trust me) and that the propper way of solving this is getting data (for now not the once I originally wanted, but hey). Now it's my time to put my nose somewhere. Diagnostic baby! Hence, a comprehensive tracking system is implemented. I allready had some mlflow tracking for generall metrics from the simulation and trainings I've done so far. This now is complemented with system monitoring and error logging (both I intended to implement, but not now.), my flight recorder, so to say.

And the answer to my problems is as simple as embarrassing: Triton works to well (and the turbo profile of my laptop only was using 90% of the fan speed)! It appears that, eventhough the vRAM was only about 50% utilized, the GPU’s processing units were under extreme strain. Triton allows us to write kernels that utilize the hardware almost perfectly. Where standard PyTorch often has pauses between operations (overhead), Triton keeps the arithmetic cores (ALUs) running at full speed without pause. We make extremely intensive use of the GPU’s fast on-chip memory (SRAM) for our 3D grid calculations. While this is fast, it generates a great deal of waste heat in a very small area of the chip. Since we calculate the nonlinear material properties (from the LUT) for each grid point, we perform a large number of mathematical operations for each loaded data point. This means the GPU cores have to “work hard” instead of just waiting for data. I monitored the GPU temperature and the base clock frequence. And sure enough, the GPU quickly reached 95°C, and the GPU frequency was throttled down to 300 MHz (this also explains the comparatively large s/it). The solution to the problem: the fan speed is cranked up to 100%, the GPU stays around 88°C (those 8°C seem to make a significant difference), and the GPU frequency stays at a slightly higher level, which in turn reduces the s/it (which is what we want). Consumer-grade hardware for highly complex technical tasks is really something special...

You might wonder: why on earth did the earlier *.gif (as can be seen on the main branches readme) simulation at even higher resolutions run through? (Likely because the complex phase-change maps were not yet active, reducing kernel branching). Why did the notebook succeed where the script initially failed? (Perhaps because the small random patches I extracted created significantly less I/O pressure, combined with shorter runs and natural 'cooldown' pauses between material runs). While I'm still narrowing down the exact tipping point, which might be a combination of I/O overhead, kernel complexity, or even something mundane as the rising ambient temperature of a sunny May afternoon, one thing is certain: actively managing the thermal ceiling (100% fan duty) was the key. Do I still need the kernel-slicing and TDR prevention now that the fans are screaming at me with a couple more dB? To be honest, I'm not entirely sure - but am I keeping it in for now? Heck yeah. These simulations take way too long and for now I'm better safe than sorry. With the system now stable at a controlled 88°C, I can finally stop debugging hardware and move on to the fun part: training and optimizing the model. 

### Lets have a quick look at the data...
Now I have my sweet data from the 10 randomized test runs, yielding 500 samples (the debugging stuff blasted my time schedule so I will add more data later if needed for this demonstration.) 

#### 1. Dataset Characteristics
| Feature | Specification |
| :--- | :--- |
| **Grid Resolution** | 512 x 256 x 64 voxels |
| **Domain Size** | 1.0 x 0.5 x 0.125 mm |
| **Voxel Size** | 1.96 µm (Isotropic) |
| **Training Samples** | 500 unique scan trajectories (Island, Raster, Zigzag) |
| **Material** | Stainless Steel 316L (Temperature-dependent properties) |

#### 2. Representative Sample Preview (Mid-Hatch State)
The following visualization (Sample 125) showcases a mid-layer state where the island scan strategy is roughly 50% complete.
![Dataset Preview](./assets/data_sample.png)

#### 3. Physics-Based Features & Normalization
To ensure training stability and physical consistency, the dataset utilizes a multi-modal feature set:
*   **Thermal Field (T):** Surface and volumetric temperature distribution (K)
*   **Phase Mapping:** A refined mapping of material states using the binary `mask` and T_solidus / T_liquidus thresholds:
    *   **Powder (Grey):** Untouched material (mask=0).
    *   **Solid (SteelBlue):** Re-solidified tracks (mask=1, T < T_solidus).
    *   **Liquid (Gold):** Active melt pool (T > T_liquidus).
*   **Normalized Heat Source (Q):** To prevent numerical saturation in float16 storage, the volumetric Gaußian heat source is normalized to a [0, 1] range relative to a global reference scale:
    *   **Reference Scale (Q_ref):** 1.35e15 W/m³
    *   **Model:** Q(r, z) = [P * eta / (2 * pi * sigma^2 * Lz)] * exp(-r^2 / 2*sigma^2)

#### 4. Material Properties (SS316L)
The simulation incorporates non-linear material behavior:
*   **Density (rho):** 7513 kg/m³
*   **Thermal Conductivity:** 13.9 W/m·K (Solid) / 33.6 W/m·K (Liquid)
*   **Latent Heat of Fusion (L):** 263 kJ/kg
*   **Solidus/Liquidus:** 1638 K / 1658 K

![T-dependent Material Parameters s316l](./assets/materials/ss316l_refined.png)

#### 5. Interpreting the Data
When taking a look at the temperature field, respectively the phase change map, you would assume that there must be, due to the extrem high cooling rates characteristic for this procedure, much more solidified areas. Because of the current state of my implementation and its simplification we basically shooting heat into an adiabatic box (Homogeneous Neumann Boundary Condition (Zero Flux), see [link to the scripts]). In other words: No heat goes out. In reality you would have a substrate plate (intenionally heated or not) where the powder is placed on, having an impact on the thermal flow. Additional cooling also happens through the shielding gas flow. In my solver right now there are now surface conditions. Furthermore the powder is assumed to be insulating due to the point contact, which is why the heat does not actually spread throughout the volume. So the heat source is heating up the exposure point, jumps to the next one and heating this one up again. The thermal energy does not disperse because there are no actual temperature gradients. But for this example, the data is sufficient for now. 

### Let's send our larger dataset through the base line
I soon aknowledged, that the gridresolution results in a way to high memory consumption, hence I decided to go again with 64x64x64 patches including the surface, cut out around the hottest area. 

![FM-Base Line Results (Surface & Depth)](./assets/FM-baseline-v3.png)

### May there be a penalty! (...a physics one)
A major advantage of this approach is its compatibility with Physics-Informed losses: since the model directly predicts the velocity field $v_t$ (the time derivative $\partial T/\partial t$), we can explicitly penalize deviations from the heat equation’s PDE residual during training. This transforms the generative model into a differentiable physics surrogate that respects conservation of energy.