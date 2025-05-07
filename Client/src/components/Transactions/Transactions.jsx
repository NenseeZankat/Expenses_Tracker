import Budget from "./Budget";

const Transactions = () => {
  return (
    <div>
      <h1>Transactions</h1>
      <Budget /> {/* Show budget management inside transactions */}
      {/* Existing transaction listing UI */}
    </div>
  );
};
export default Transactions;
