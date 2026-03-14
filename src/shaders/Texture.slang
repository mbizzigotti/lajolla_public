import shared;
import Scene;

float3 eval(int texture_id, float2 uv, float footprint)
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
        if (t.index != 0)
            printf("index = %i", t.index);

		int width;
		int height;
		scene.textures[t.index].GetDimensions(width, height);
		
		float scaled_footprint = float(max(width, height)) * max(t.scale.x, t.scale.y) * footprint;
        float level = log2(max(scaled_footprint, 1e-8));
        return scene.textures[t.index].SampleLevel(scene.sampler, local_uv, level).rgb;
	}
}

float3 eval(float3_parameter parameter, float2 uv, float footprint)
{
	if (parameter.texture_id != -1)
    {
        return eval(parameter.texture_id, uv, footprint);
    }
    return parameter.value;
}
