import { createSlice } from '@reduxjs/toolkit';

const userSlice = createSlice({
  name: 'user',
  initialState: {
    profilePic: '/default-profile.png',
    user: null, // Stores full user object if needed
  },
  reducers: {
    setProfilePic: (state, action) => {
      state.profilePic = action.payload;
    },
    setUser: (state, action) => {
      state.user = action.payload; // Full user object
      if (action.payload?.profilePic) {
        state.profilePic = `https://backend-johh.onrender.com${action.payload.profilePic}`;
      }
    },
  },
});

export const { setProfilePic, setUser } = userSlice.actions;

export default userSlice.reducer;
