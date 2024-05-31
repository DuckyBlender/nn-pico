struct XorshiftRng {
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

impl XorshiftRng {
    fn next_u32(&mut self) -> u32 {
        let t = self.x ^ (self.x << 11);
        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        self.w = self.w ^ (self.w >> 19) ^ (t ^ (t >> 8));
        self.w
    }
}

static mut XORSHIFT_RNG: XorshiftRng = XorshiftRng {
    x: 123456789,
    y: 362436069,
    z: 521288629,
    w: 88675123,
};

fn random_f32() -> f32 {
    unsafe {
        let random_u32 = XORSHIFT_RNG.next_u32();
        let random_f32 = (random_u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        random_f32
    }
}