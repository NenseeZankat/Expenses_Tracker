const cron = require("node-cron");
const Transaction = require("../model/Transaction");
const User = require("../model/User");
const Budget = require("../model/Budget"); // New model for budget tracking
const FamilyGoal = require("../model/FamilyGoal"); // Family goal model

// Schedule job to run at midnight on the first day of every month
cron.schedule("0 0 1 * *", async () => {
    console.log("Running monthly budget allocation...");

    try {
        const users = await User.find(); // Fetch all users

        for (const user of users) {
            const { _id, allocationPreference } = user;
            const lastMonthTransactions = await Transaction.find({ userId: _id, month: getLastMonth() });

            const totalIncome = lastMonthTransactions
                .filter(txn => txn.type === "income")
                .reduce((sum, txn) => sum + txn.amount, 0);

            const totalExpense = lastMonthTransactions
                .filter(txn => txn.type === "expense")
                .reduce((sum, txn) => sum + txn.amount, 0);

            const leftoverAmount = totalIncome - totalExpense;

            if (leftoverAmount > 0) {
                if (allocationPreference === "next_month") {
                    await Budget.findOneAndUpdate(
                        { userId: _id, month: getNextMonth() },
                        { $inc: { amount: leftoverAmount } },
                        { upsert: true, new: true }
                    );
                } else if (allocationPreference === "family_goal") {
                    await FamilyGoal.findOneAndUpdate(
                        { userId: _id },
                        { $inc: { savedAmount: leftoverAmount } },
                        { upsert: true, new: true }
                    );
                }
            }
        }

        console.log("Budget allocation completed.");
    } catch (error) {
        console.error("Error in budget allocation:", error);
    }
});

// Helper functions
function getLastMonth() {
    const now = new Date();
    const lastMonth = new Date(now.getFullYear(), now.getMonth() - 1, 1);
    return `${lastMonth.getFullYear()}-${String(lastMonth.getMonth() + 1).padStart(2, "0")}`;
}

function getNextMonth() {
    const now = new Date();
    const nextMonth = new Date(now.getFullYear(), now.getMonth() + 1, 1);
    return `${nextMonth.getFullYear()}-${String(nextMonth.getMonth() + 1).padStart(2, "0")}`;
}

module.exports = cron;
