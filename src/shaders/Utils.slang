
[ForceInline]
float luminance(float3 s)
{
    return dot(s, float3(0.212671, 0.715160, 0.072169));
}

[ForceInline]
float3 xform_point(float4x4 xform, float3 pt) {
    float4 tpt = mul(transpose(xform), float4(pt, 1));
    float inv_w = 1 / tpt[3];
    return float3(tpt[0] * inv_w, tpt[1] * inv_w, tpt[2] * inv_w);
}

[ForceInline]
float3 xform_vector(float4x4 xform, float3 vec) {
    return mul(transpose(xform), float4(vec, 0)).xyz;
}

[ForceInline]
float3 sample_cos_hemisphere(float2 rnd_param)
{
	float phi = 2 * float.getPi() * rnd_param[0];
    float tmp = sqrt(saturate(1 - rnd_param[1]));
    return {
        cos(phi) * tmp, sin(phi) * tmp,
        sqrt(saturate(rnd_param[1]))
    };
}

[ForceInline]
float3 sample_half_vector(float alpha, float2 rnd_param)
{
    float a2 = alpha * alpha;
    float cos_h_elevation = sqrt(
        (float(1) - pow(a2, float(1) - rnd_param.x))
      / (float(1) - a2));
    float h_azimuth = 2 * float.getPi() * rnd_param.y;
    float sin_h_azimuth = sin(h_azimuth);
    float cos_h_azimuth = cos(h_azimuth);
    float sin_h_elevation = sqrt(float(1) - cos_h_elevation * cos_h_elevation);
    return float3(
        sin_h_elevation * cos_h_azimuth,
        sin_h_elevation * sin_h_azimuth,
        cos_h_elevation
    );
}
