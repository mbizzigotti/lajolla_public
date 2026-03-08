import shared;
import random;
import Utils;

extension Filter
{
	float2 sample(float2 rnd_param)
	{
        switch (type)
        {
		default:
        case FilterType.Box:
        {
            return (2 * rnd_param - 1) * (value / 2);
		}
        case FilterType.Tent:
        {
			float h = value / 2;
            float x = rnd_param[0] < 0.5 ? h * (sqrt(2 * rnd_param[0]) - 1) : h * (1 - sqrt(1 - 2 * (rnd_param[0] - 0.5)));
            float y = rnd_param[1] < 0.5 ? h * (sqrt(2 * rnd_param[1]) - 1) : h * (1 - sqrt(1 - 2 * (rnd_param[1] - 0.5)));
            return float2(x, y);
        }
        case FilterType.Gaussian:
        {
            float r = value * sqrt(-2 * log(max(rnd_param[0], 1e-8)));
            return { r * cos(2 * float.getPi() * rnd_param[1]),
                     r * sin(2 * float.getPi() * rnd_param[1]) };
        }
		}
	}
};

extension Camera
{
    RayDesc sample_primary(uint2 pixel_pos, uint2 dim, inout pcg32_state rng)
	{
        // Importance sample from the pixel filter (see filter.h for more explanation).
        // We first extract the subpixel offset.
        float dx = rng.next_float();
        float dy = rng.next_float();
        // dx/dy are uniform variables in [0, 1],
        // so let's remap them using importance sampling.
        float2 offset = 0.5 * filter.sample(float2(dx, dy));
        // Filters are placed at pixel centers.
        float2 remapped_pos = (float2(pixel_pos) + 0.5 + offset) / float2(dim);
        float3 pt = xform_point(sample_to_cam, float3(remapped_pos.xy, 0));
        float3 dir = normalize(pt);
        RayDesc ray;
        ray.Origin = xform_point(cam_to_world, float3(0, 0, 0));
        ray.Direction = normalize(xform_vector(cam_to_world, dir));
        ray.TMin = 0;
        ray.TMax = 1e9;
        return ray;
	}
};
