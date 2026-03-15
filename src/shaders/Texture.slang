import Shared;
import Scene;

float3 eval3(int texture_id, float2 uv, float footprint)
{
	TextureInfo t = scene.texture_infos[texture_id];
	float2 local_uv = fmod(uv * t.scale + t.offset, 1);

    if (bool(t.is_checkerboard))
    {
        int x = 2 * int(local_uv.x * 2).mod(2) - 1;
		int y = 2 * int(local_uv.y * 2).mod(2) - 1;
        return x * y == 1 ? t.color0 : t.color1;
    }
    else
    {
		int width;
		int height;
		scene.textures[t.index].GetDimensions(width, height);
		
		float scaled_footprint = float(max(width, height)) * max(t.scale.x, t.scale.y) * footprint;
        float level = log2(max(scaled_footprint, 1e-8));
        return scene.textures[t.index].SampleLevel(scene.sampler, local_uv, level).rgb;
	}
}

float3 eval3(Parameter parameter, float2 uv, float footprint)
{
	if (parameter.texture_id != -1)
    {
        return eval3(parameter.texture_id, uv, footprint);
    }
    return parameter.value;
}

float eval1(int texture_id, float2 uv, float footprint)
{
    TextureInfo t = scene.texture_infos[texture_id];
    float2 local_uv = fmod(uv * t.scale + t.offset, 1);

    if (bool(t.is_checkerboard))
    {
        int x = 2 * int(local_uv.x * 2).mod(2) - 1;
        int y = 2 * int(local_uv.y * 2).mod(2) - 1;
        return x * y == 1 ? t.color0.x : t.color1.x;
    }
    else
    {
        int width;
        int height;
        scene.textures[t.index].GetDimensions(width, height);

        float scaled_footprint = float(max(width, height)) * max(t.scale.x, t.scale.y) * footprint;
        float level = log2(max(scaled_footprint, 1e-8));
        return scene.textures[t.index].SampleLevel(scene.sampler, local_uv, level).r;
    }
}

float eval1(Parameter parameter, float2 uv, float footprint)
{
    if (parameter.texture_id != -1)
    {
        return eval1(parameter.texture_id, uv, footprint);
    }
    return parameter.value.x;
}
