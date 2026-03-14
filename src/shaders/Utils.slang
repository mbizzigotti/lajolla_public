
[ForceInline]
float distance_squared(float3 a, float3 b)
{
    float3 diff = b - a;
    return dot(diff, diff);
}

[ForceInline]
float luminance(float3 s)
{
    return dot(s, float3(0.212671, 0.715160, 0.072169));
}

[ForceInline]
func max_comp(float3 v) -> float
{
    return max(v.x, max(v.y, v.z));
}

[ForceInline]
float3 xform_point(float4x4 xform, float3 pt)
{
    float4 tpt = mul(transpose(xform), float4(pt, 1));
    float inv_w = 1 / tpt[3];
    return float3(tpt[0] * inv_w, tpt[1] * inv_w, tpt[2] * inv_w);
}

[ForceInline]
float3 xform_vector(float4x4 xform, float3 vec)
{
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

int upper_bound(ByteAddressBuffer arr, int offset, int count, float value)
{
    int first = offset;
    int count = count;
    while (0 < count) { // divide and conquer, find half that contains answer
		int count2 = count / 2;
        int mid = first + count2;
        float mid_value = arr.LoadByteOffset<float>(mid * sizeof(float));
        if (value < mid_value) {
            count = count2;
        } else { // try top half
            first = mid + 1;
            count -= count2 + 1;
        }
    }
    return first;
}
