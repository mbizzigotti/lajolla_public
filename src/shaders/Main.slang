import shared;
import Scene;
import Camera;
import random;
import Intersection;
import Light;

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
		float light_rnd = rng.next_float();
        int light_id = sample_light(light_rnd);
		float3 light_sample_rnd = rng.next_float3();
        Light light = scene.lights[light_id];
        PointAndNormal point_on_light = sample_point_on_light(light, vertex.position, light_sample_rnd);

        float3 dir_light = normalize(point_on_light.position - vertex.position);
		#if 1
        RayDesc shadow_ray;
        shadow_ray.Origin = vertex.position;
        shadow_ray.Direction = dir_light;
        shadow_ray.TMin = 0.01;
        shadow_ray.TMax = (1 - 0.01) * distance(point_on_light.position, vertex.position);
        if (!occluded(shadow_ray))
        {
            let material = scene.materials[vertex.material_id];
			result = float4(material.reflectance, 1);
        }
		#else
        result = float4(dir_light, 1);
		#endif
	}
    scene.accumulate_sample(pixel_index, result);
}
