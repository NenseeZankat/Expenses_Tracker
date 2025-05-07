import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Completion() {
  const [transactionData, setTransactionData] = useState(null);
  console.log(transactionData);
  const navigate = useNavigate();

  useEffect(() => {
    const data = JSON.parse(localStorage.getItem("transactionData"));
    if (data) {
      setTransactionData(data);
    } else {
      navigate("/add-purchase");
    }
  }, [navigate]);

  if (!transactionData) {
    return <div className="text-center mt-10 text-gray-600">Loading transaction details...</div>;
  }

  return (
    <div className="max-w-lg mx-auto my-10 bg-white p-6 rounded-lg shadow-lg">
      <h2 className="text-2xl font-semibold text-gray-800 text-center">Transaction Complete</h2>
      <p className="text-gray-600 text-center">Here are your transaction details:</p>

      <div className="mt-6 space-y-4">
        <div>
          <strong>Amount:</strong> ${transactionData.amount}
        </div>
        <div>
          <strong>Category:</strong> {transactionData.category}
        </div>
        <div>
          <strong>Type:</strong> {transactionData.type}
        </div>
        <div>
          <strong>Location:</strong> {transactionData.placeName || "Not provided"}
        </div>
      </div>

      <button
        onClick={() => navigate("/add-purchase")}
        className="mt-6 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded w-full"
      >
        Back to Home
      </button>
    </div>
  );
}
