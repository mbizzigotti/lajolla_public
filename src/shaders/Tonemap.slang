
Texture2D<float4> input;
RWTexture2D<float4> output;

[numthreads(8, 8, 1)]
[shader("compute")]
void tonemap(uint3 tid : SV_DispatchThreadID, uniform float exposure)
{
    float3 hdr = input[tid.xy].rgb;

	hdr *= exposure;
    float3 mapped = hdr / (1.0 + hdr); // Reinhard tone mapping
    float3 gamma = pow(mapped, 1.0 / 2.2); // gamma correction

    output[tid.xy] = float4(gamma, 1.0);
}
