import Shared;
import Scene;

float3 lookup(in Volume volume, float3 position)
{
    if (volume.index == -1)
    {
        return volume.max_data;
	}

    float3 texcoord = (position - volume.p_min) / (volume.p_max - volume.p_min);
    return scene.volumes[volume.index].SampleLevel(scene.volume_sampler, texcoord, 0);
}

float3 get_sigma_a(in Medium medium, float3 p)
{
    if (!bool(medium.is_heterogenious))
        return medium.albedo.max_data;

    float3 albedo = lookup(medium.albedo, p);
    float3 density = lookup(medium.density, p);
    return density * (1 - albedo);
}

float3 get_sigma_s(in Medium medium, float3 p)
{
    if (!bool(medium.is_heterogenious))
        return medium.density.max_data;

    float3 albedo = lookup(medium.albedo, p);
    float3 density = lookup(medium.density, p);
    return density * albedo;
}

bool intersect(in Volume v, in RayDesc ray)
{
    // https://github.com/mmp/pbrt-v3/blob/master/src/core/geometry.h#L1388
    float t0 = 0, t1 = ray.TMax;
    for (int i = 0; i < 3; i++) {
        float tnear = (v.p_min[i] - ray.Origin[i]) / ray.Direction[i];
        float tfar = (v.p_max[i] - ray.Origin[i]) / ray.Direction[i];

        // Update parametric interval from slab intersection $t$ values
        if (tnear > tfar) {
            float tmp = tfar;
            tfar = tnear;
            tnear = tmp;
        }

        t0 = tnear > t0 ? tnear : t0;
        t1 = tfar < t1 ? tfar : t1;
        if (t0 > t1) {
            return false;
        }
    }
    return true;
}

float3 get_majorant(in Medium medium, in RayDesc ray)
{
    if (!bool(medium.is_heterogenious))
    {
		return medium.albedo.max_data + medium.density.max_data;
    }
    if (intersect(medium.density, ray))
	{
        return medium.density.max_data;
    }
    return 0;
}
