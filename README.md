# GeoInvFlow
Geometry and physics-aware normalizing flows for structurally non-identifiable inverse problems

## Quadratic inverse experiment

Given a scalar observation $y$, the hybrid conditional normalizing flow samples input pairs $(x_1, x_2)$ that approximately reproduce $y$ under the quadratic forward map. The training objective combines negative log-likelihood with a reconstruction loss. This loss measures the mean squared error between the conditioning observation and the output of a separately trained, fixed forward model evaluated on each generated input pair.

The hybrid CNF decodes 500 Gaussian samples at each observation value. The animation shows the decoded inputs against the exact level set and on the forward surface.

![Hybrid CNF animation for the quadratic inverse experiment](Experiments/Quadratic/outputs/hybrid/conditional_inverse.gif)
