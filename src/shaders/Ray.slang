#pragma once

/// We adopt an approach for ray differentials
/// that is used in Renderman.
/// See Section 6.6 in "RenderMan: 
/// An Advanced Path Tracing Architecture for Movie Rendering"
/// https://graphics.pixar.com/library/RendermanTog2018/paper.pdf
/// The idea is to simplify Igehy's ray differential by only
/// storing two quantities: a "radius" that describes positional
/// differential, and a "spread" that describes directional differential.

/// For glossy/diffuse surfaces, Renderman used a heuristics based on the
/// PDF of the sampling direction and use larger spread for low PDF.
/// Here we use an even simpler heuristics: we linearly blend
/// between the specular spread and a constant based on roughness.
struct RayDifferential
{
    // Radius is approximately (length(dp/dx) + length(dp/dy)) / 2
    // Spread is approximately (length(dd/dx) + length(dd/dy)) / 2
    // p is ray position, d is ray direction.
    float radius, spread; // The units are pixels.
};

/// These functions propagate the ray differential information.
RayDifferential init_ray_differential(uint2 dim)
{
    return RayDifferential(0, 0.25 / float(max(dim.x, dim.y)));
}

/// Update the radius (dp/dx) of a ray differential by propagating it over a distance.
[ForceInline]
float transfer(in RayDifferential r, float dist)
{
    return r.radius + r.spread * dist;
}

/// Update the spread (dd/dx) of a ray differential by scattering over a reflective surface.
[ForceInline]
float reflect(in RayDifferential r, float mean_curvature, float roughness)
{
    float spec_spread = r.spread + 2 * mean_curvature * r.radius;
    float diff_spread = 0.2;
    return max(spec_spread * (1 - roughness) + diff_spread * roughness, 0);
}

/// Update the spread (dd/dx) of a ray differential by scattering over a refractive surface.
/// The Renderman reference https://graphics.pixar.com/library/RendermanTog2018/paper.pdf
/// did not specify how they propagate ray differentials through the refraction.
/// We simply guess a formula here: when eta == 1, the spread & radius should not change.
/// High eta makes ray more concentrated and reduces the spread, and vice versa.
/// so we simply divide everything by eta.
[ForceInline]
float refract(in RayDifferential r, float mean_curvature, float eta, float roughness)
{
    float spec_spread = (r.spread + 2 * mean_curvature * r.radius) / eta;
    float diff_spread = 0.2;
    return max(spec_spread * (1 - roughness) + diff_spread * roughness, 0);
}
