
struct pcg32_state
{
    uint64_t state;
    uint64_t inc;
	
	[mutating]
	uint32_t next() {
		uint64_t oldstate = state;
		// Advance internal state
		state = oldstate * 6364136223846793005ull + (inc|1);
		// Calculate output function (XSH RR), uses old state for max ILP
		uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
		uint32_t rot = uint32_t(oldstate >> 59u);
		return uint32_t((xorshifted >> rot) | (xorshifted << ((-rot) & 31)));
    }

    [mutating]
    float next_float()
    {
        uint32_t u = (next() >> 9) | 0x3f800000u;
        return bit_cast<float>(u) - 1.0;
    }

    [mutating]
    float2 next_float2()
    {
        float x = next_float();
        float y = next_float();
        return float2(x, y);
    }

    [mutating]
    float3 next_float3()
    {
        float x = next_float();
        float y = next_float();
        float z = next_float();
        return float3(x, y, z);
    }
};

pcg32_state init_pcg32(uint64_t stream_id = 1, uint64_t seed = 0x31e241f862a1fb5eULL)
{
    pcg32_state s;
    s.state = 0U;
    s.inc = (stream_id << 1u) | 1u;
    s.next();
    s.state += seed;
    s.next();
    return s;
}
