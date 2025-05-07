// import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
// import axios from "axios"; // Ensure you have Axios setup

// // ✅ Fetch budget
// export const fetchBudget = createAsyncThunk(
//   "budget/fetch",
//   async (month, { rejectWithValue }) => {
//     try {
//       const response = await axios.get(`/api/budgets?month=${month}`);
//       return response.data;
//     } catch (error) {
//       return rejectWithValue(error.response.data);
//     }
//   }
// );


// export const setBudget = createAsyncThunk(
//     "budget/set",
//     async ({ month, allocatedBudget }, { rejectWithValue }) => {
//       try {
//         const token = localStorage.getItem("token"); 
//         console.log(token);// Ensure token is stored in localStorage
//         const config = {
//           headers: {
//             Authorization: `Bearer ${token}`, 
//           },
//         };
  
//         const response = await axios.post("http://localhost:8000/api/budgets/create", 
//           { month, allocatedBudget }, 
//           config
//         );
//         console.log("Budget API Response:", response.data); // ✅ Log API response
//         return response.data;
//       } catch (error) {
//         return rejectWithValue(error.response.data || "Server error");
//       }
//     }
//   );
  

//   export const updateBudget = createAsyncThunk(
//     "budget/update",
//     async ({ month, amount }, { rejectWithValue }) => {
//       try {
//         const token = localStorage.getItem("token"); 
//         const config = {
//           headers: {
//             Authorization: `Bearer ${token}`,
//           },
//         };
  
//         const response = await axios.put("http://localhost:8000/api/budgets/update", 
//           { month, amount }, 
//           config
//         );
  
//         return response.data;
//       } catch (error) {
//         return rejectWithValue(error.response.data || "Server error");
//       }
//     }
//   );
  
// const budgetSlice = createSlice({
//   name: "budget",
//   initialState: { budget: null, loading: false, error: null },
//   reducers: {},
//   extraReducers: (builder) => {
//     builder
//       .addCase(fetchBudget.pending, (state) => {
//         state.loading = true;
//       })
//       .addCase(fetchBudget.fulfilled, (state, action) => {
//         state.loading = false;
//         state.budget = action.payload;
//       })
//       .addCase(fetchBudget.rejected, (state, action) => {
//         state.loading = false;
//         state.error = action.payload;
//       })
//       .addCase(setBudget.fulfilled, (state, action) => {
//         state.budget = action.payload;
//       })
//       .addCase(updateBudget.fulfilled, (state, action) => {  // ✅ Add this case
//         state.budget = action.payload;
//       });
//   },
// });





// export default budgetSlice.reducer;


import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import axios from "axios"; // Ensure Axios is installed

// ✅ Fetch budget
export const fetchBudget = createAsyncThunk(
  "budget/fetch",
  async (month, { rejectWithValue }) => {
    try {
      const response = await axios.get(`/api/budgets?month=${month}`);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || "Server error");
    }
  }
);

// ✅ Set budget
export const setBudget = createAsyncThunk(
  "budget/set",
  async ({ month, allocatedBudget }, { rejectWithValue }) => {
    try {
      const token = localStorage.getItem("token");
      const config = {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      };

      const response = await axios.post(
        "https://backend-johh.onrender.com/api/budgets/create",
        { month, allocatedBudget },
        config
      );
      console.log("Budget API Response:",token);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || "Server error");
    }
  }
);



export const updateBudget = createAsyncThunk(
  "budget/update",
  async ({ month,allocatedBudget}, { rejectWithValue }) => {
    try {
      const token = localStorage.getItem("token");
      // console.log("Update Budget Request:", { month,allocatedBudget });
      // console.log("Token:", token);

      const config = {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      };

      const response = await axios.put(
        "https://backend-johh.onrender.com/api/budgets/update",
        { month, amount: allocatedBudget},
        config
      );

      console.log("Update Budget Response:", response.data);
      return response.data;
    } catch (error) {
      console.error("Update Budget Error:", error.response?.data || error.message);
      return rejectWithValue(error.response?.data || "Server error");
    }
  }
);


// ✅ Create slice with all reducers
const budgetSlice = createSlice({
  name: "budget",
  initialState: { budget: null, loading: false, error: null },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchBudget.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchBudget.fulfilled, (state, action) => {
        state.loading = false;
        state.budget = action.payload;
      })
      .addCase(fetchBudget.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })
      .addCase(setBudget.fulfilled, (state, action) => {
        state.budget = action.payload;
      })
      .addCase(updateBudget.fulfilled, (state, action) => { // ✅ Now it is recognized
        state.budget = action.payload;
      });
  },
});

export default budgetSlice.reducer;
