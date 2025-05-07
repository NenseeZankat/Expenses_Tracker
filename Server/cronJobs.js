// const cron = require("node-cron");
// const Subscription = require("./model/Subscription");
// const moment = require("moment");
// const sendNotification = require("./Services/notificationService");

// // cron.schedule("* * * * *", async () => { // Runs daily at 9 AM
  
// async function runJob() {
//     console.log("Running Subscription Alert Job");

//   const today = moment().startOf("day");
//   const subscriptions = await Subscription.find({ notified: false });
//   console.log("Subscriptions found:", subscriptions.length); // 🔍 Debugging log

//   subscriptions.forEach(async (sub) => {
//     const renewalDate = moment(sub.renewalDate).startOf("day");
//     const alertDate = renewalDate.subtract(sub.alertDaysBefore, "days");
//     console.log("Checking:", sub.serviceName, "Alert Date:", alertDate.format("YYYY-MM-DD"), "Today:", today.format("YYYY-MM-DD")); // 🔍 Debugging log

//     if (today.isSame(alertDate, "day")) {
//         console.log("Sending notification for:", sub.serviceName); // 🔍 Debugging log
//       await sendNotification(sub.userId, `Your subscription for ${sub.serviceName} is renewing soon!`);
//       await Subscription.findByIdAndUpdate(sub._id, { notified: true });
//     }
//    });
//  }
// // );
// runJob();

// cron.schedule("0 9 * * *", async () => {
//     await runJob();
// });


const cron = require("node-cron");
const Subscription = require("./model/Subscription");
const moment = require("moment");
const sendNotification = require("./Services/notificationService");

async function runJob() {
    console.log("Running Subscription Alert Job");

    const today = moment().startOf("day");
    const subscriptions = await Subscription.find({ notify: true }); // Find subscriptions where notification is enabled

    console.log("Subscriptions found:", subscriptions.length);

    for (const sub of subscriptions) {
        const renewalDate = moment(sub.renewalDate).startOf("day");
        const alertDate = renewalDate.clone().subtract(sub.alertDaysBefore, "days");

        console.log("Checking:", sub.serviceName, "Alert Date:", alertDate.format("YYYY-MM-DD"), "Today:", today.format("YYYY-MM-DD"));

        if (today.isSame(alertDate, "day")) {
            console.log("Sending notification for:", sub.serviceName);
            await sendNotification(sub.userId, `Your subscription for ${sub.serviceName} is renewing soon!`);

            // ✅ Update the next renewal date based on frequency (monthly, yearly, etc.)
            let nextRenewalDate = renewalDate.clone();
            if (sub.frequency === "monthly") {
                nextRenewalDate.add(1, "month");
            } else if (sub.frequency === "yearly") {
                nextRenewalDate.add(1, "year");
            }

            // ✅ Reset `notified` and update the renewal date
            await Subscription.findByIdAndUpdate(sub._id, { 
                renewalDate: nextRenewalDate, 
                notified: false 
            });
        }
    }
}

// Run job immediately for testing
runJob();

// Schedule job to run daily at 9 AM
cron.schedule("0 9 * * *", async () => {
    await runJob();
});
