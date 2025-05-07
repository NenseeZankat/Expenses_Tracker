import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import axios from "axios";

// ✅ Fetch Debt Details
export const fetchDebt = createAsyncThunk("debt/fetchDebt", async (_, { rejectWithValue }) => {
  try {
    const token = localStorage.getItem("token");
    const response = await axios.get("https://backend-johh.onrender.com/api/debt/get", {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data;
  } catch (error) {
    return rejectWithValue(error.response?.data || "Error fetching debt details");
  }
});

// ✅ Add New Debt
export const addDebt = createAsyncThunk("debt/addDebt", async ({ amount,loanPurpose, interestRate, monthlyEMI }, { rejectWithValue }) => {
  try {
    const token = localStorage.getItem("token");
    const response = await axios.post(
      "https://backend-johh.onrender.com/api/debt/add",
      { amount,loanPurpose, interestRate, monthlyEMI },
      { headers: { Authorization: `Bearer ${token}` } }
    );
    return response.data;
  } catch (error) {
    return rejectWithValue(error.response?.data || "Error adding debt");
  }
});

// ✅ Make a Payment
export const payEMI = createAsyncThunk("debt/payEMI", async ({ debtId,amount}, { rejectWithValue }) => {
  try {
    const token = localStorage.getItem("token");
    const response = await axios.post(
      "https://backend-johh.onrender.com/api/debt/pay",
      { debtId, amount },
      { headers: { Authorization: `Bearer ${token}` } }
    );
    return response.data;
  } catch (error) {
    return rejectWithValue(error.response?.data || "Error making payment");
  }
});

export const deleteDebt = createAsyncThunk("debt/deleteDebt", async (debtId, { rejectWithValue }) => {
    try {
      const token = localStorage.getItem("token");
      await axios.delete(`https://backend-johh.onrender.com/api/debt/delete/${debtId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      return debtId; // Returning the ID so we can remove it from the Redux store
    } catch (error) {
      return rejectWithValue(error.response?.data || "Error deleting debt");
    }
  });
// const debtSlice = createSlice({
//   name: "debt",
//   initialState: { debt: null, loading: false, error: null },
//   reducers: {},
//   extraReducers: (builder) => {
//     builder
//       .addCase(fetchDebt.pending, (state) => {
//         state.loading = true;
//       })
//       .addCase(fetchDebt.fulfilled, (state, action) => {
//         state.loading = false;
//         state.debt = action.payload;
//       })
//       .addCase(fetchDebt.rejected, (state, action) => {
//         state.loading = false;
//         state.error = action.payload;
//       })
//       .addCase(addDebt.fulfilled, (state, action) => {
//         state.debt = action.payload;
//       })
//       .addCase(payEMI.fulfilled, (state, action) => {
//         state.debt = action.payload;
//       })
//       .addCase(deleteDebt.fulfilled, (state, action) => {
//         state.debt = state.debt.filter((debt) => debt.id !== action.payload);
//       });
//   },
// });

// export default debtSlice.reducer;

const debtSlice = createSlice({
    name: "debt",
    initialState: { debts: [], loading: false, error: null }, // ✅ Ensure `debts` is an array
    reducers: {},
    extraReducers: (builder) => {
      builder
        .addCase(fetchDebt.pending, (state) => {
          state.loading = true;
        })
        .addCase(fetchDebt.fulfilled, (state, action) => {
          state.loading = false;
          state.debts = action.payload; // ✅ Ensure correct property name
        })
        .addCase(fetchDebt.rejected, (state, action) => {
          state.loading = false;
          state.error = action.payload;
        })
        .addCase(addDebt.fulfilled, (state, action) => {
          state.debts.push(action.payload);
        })
        .addCase(deleteDebt.fulfilled, (state, action) => {
          state.debts = state.debts.filter((debt) => debt._id !== action.payload);
        })
        .addCase(payEMI.fulfilled, (state, action) => {
          const updatedDebt = action.payload; // API should return full updated debt object
          state.debts = state.debts.map((debt) =>
            debt._id === updatedDebt._id ? updatedDebt : debt // ✅ Replace old debt with updated debt
          );
        });
      
    },
  });
  
  export default debtSlice.reducer;
  