import shared;
import Scene;

SamplerState sampler;
Texture2D textures[];

float3 eval(int texture_id, float2 uv, float footprint)
{
    return 0;
	#if 0
    int width;
    int height;
    textures[texture_id].GetDimensions(width, height);
    
	TextureInfo t = scene.texture_infos[texture_id];
    float2 local_uv = {(uv[0] * t.uscale + t.uoffset).mod(1),
                       (uv[1] * t.vscale + t.voffset).mod(1)};
    float scaled_footprint = float(max(width, height)) * max(t.uscale, t.vscale) * footprint;
    float level = log2(max(scaled_footprint, 1e-8));
   	return textures[texture_id].SampleLevel(sampler, local_uv, level).rgb;
	#endif
}
