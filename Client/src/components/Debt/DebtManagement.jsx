import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { fetchDebt, payEMI } from "../../redux/slice/debtSlice";

const DebtManagement = () => {
  const dispatch = useDispatch();
  const { debt, loading, error } = useSelector((state) => state.debt);
  const [paymentAmount, setPaymentAmount] = useState("");

  useEffect(() => {
    dispatch(fetchDebt());
  }, [dispatch]);

  const handlePayment = () => {
    if (!paymentAmount || paymentAmount <= 0) {
      alert("Enter a valid amount");
      return;
    }
    dispatch(payEMI(paymentAmount));
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold">Debt Management</h2>

      {loading ? (
        <p>Loading...</p>
      ) : error ? (
        <p className="text-red-500">{error}</p>
      ) : debt ? (
        <div className="mt-4">
          <p>Total Loan: ₹{debt.amount}</p>
          <p>Interest Rate: {debt.interestRate}%</p>
          <p>Monthly EMI: ₹{debt.monthlyEMI}</p>
          <p>Total Interest Paid: ₹{debt.totalInterestPaid}</p>
          <p>Remaining Loan: ₹{debt.remainingAmount}</p>

          <input
            type="number"
            placeholder="Enter Payment Amount"
            value={paymentAmount}
            onChange={(e) => setPaymentAmount(e.target.value)}
            className="border p-2 rounded w-full mt-2"
          />
          <button onClick={handlePayment} className="bg-blue-500 text-white p-2 rounded mt-2">
            Pay EMI
          </button>
        </div>
      ) : (
        <p>No active debts found</p>
      )}
    </div>
  );
};

export default DebtManagement;
