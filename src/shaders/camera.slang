#define PI 3.141592653589793238462643383279502884197169

#include "data_structures.inl"

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
            return { r * cos(2 * PI * rnd_param[1]),
                     r * sin(2 * PI * rnd_param[1]) };
        }
		}
	}
};

float3 xform_point(float4x4 xform, float3 pt) {
    float4 tpt = mul(transpose(xform), float4(pt, 1));
    float inv_w = 1 / tpt[3];
    return float3(tpt[0] * inv_w, tpt[1] * inv_w, tpt[2] * inv_w);
}

float3 xform_vector(float4x4 xform, float3 vec) {
    return mul(transpose(xform), float4(vec, 0)).xyz;
}

extension Camera
{
    RayDesc sample_primary(float2 screen_pos)
	{
        // screen_pos' domain is [0, 1]^2
        float2 pixel_pos = { screen_pos.x * width, screen_pos.y * height };

        // Importance sample from the pixel filter (see filter.h for more explanation).
        // We first extract the subpixel offset.
        float dx = pixel_pos.x - floor(pixel_pos.x);
        float dy = pixel_pos.y - floor(pixel_pos.y);
        // dx/dy are uniform variables in [0, 1],
        // so let's remap them using importance sampling.
        float2 offset = filter.sample(float2(dx, dy));
        // Filters are placed at pixel centers.
        float2 remapped_pos = {
            (floor(pixel_pos.x) + 0.5 + offset.x) / width,
            (floor(pixel_pos.y) + 0.5 + offset.y) / height
		};

        float3 pt = xform_point(sample_to_cam,
                                 float3(remapped_pos[0], remapped_pos[1], 0));
        float3 dir = normalize(pt);
        RayDesc ray;
        ray.Origin = xform_point(cam_to_world, float3(0, 0, 0));
        ray.Direction = normalize(xform_vector(cam_to_world, dir));
        ray.TMin = 0;
        ray.TMax = 1e9;
        return ray;
	}
};
