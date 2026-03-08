import Scene;
import Camera;
import random;
import Intersection;

import Utils;
import Frame;

// ------------------------------------------------------------
// Ray Generation Shader (Path Tracing)
// ------------------------------------------------------------

[shader("raygeneration")]
void rgen_path_tracing()
{
    uint2 pixel_index = DispatchRaysIndex().xy;
    uint2 dimensions = DispatchRaysDimensions().xy;
    pcg32_state rng = init_pcg32(frame.index * (pixel_index.y * dimensions.x + pixel_index.x));
    RayDesc ray = scene.camera.sample_primary(pixel_index, dimensions, rng);
	float4 result = float4(0);
	if (let vertex = intersect(ray))
    {
        float3 dir = to_world(vertex.shading_frame, sample_cos_hemisphere(rng.next_float2()));
        RayDesc shadow_ray;
        shadow_ray.Origin = vertex.position;
        shadow_ray.Direction = dir;//float3(0, 1, 0);
        shadow_ray.TMin = 0.01;
        shadow_ray.TMax = 10.0;
        if (!occluded(shadow_ray))
        {
			result = float4(normalize(vertex.position), 1);
		}
	}
    scene.accumulate_sample(pixel_index, result);
}
