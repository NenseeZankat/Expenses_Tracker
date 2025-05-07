

// const fs = require("fs");
// const path = require("path");
// const PDFDocument = require("pdfkit");
// const Transaction = require("../model/Transaction");

// exports.generateTaxReport = async (req, res) => {
//     try {
//         const { userId, year } = req.params;
//         console.log("Generating report for:", userId, "Year:", year);

//         // Fetch transactions for the given user and year
//         const transactions = await Transaction.find({
//             user: userId,  
//             date: { $gte: new Date(`${year}-01-01`), $lt: new Date(`${year}-12-31`) }
//         });

//         console.log("Transactions Found:", transactions);

//         if (!transactions.length) {
//             return res.status(404).json({ message: "No transactions found for this user and year." });
//         }

//         // Define the directory and file path
//         const reportsDir = path.join(__dirname, "../reports");
//         const filePath = path.join(reportsDir, `tax-report-${userId}-${year}.pdf`);

//         console.log("PDF will be saved at:", filePath);

//         // ✅ Ensure the `reports` directory exists
//         if (!fs.existsSync(reportsDir)) {
//             console.log("Creating reports directory...");
//             fs.mkdirSync(reportsDir, { recursive: true });
//             console.log("Created reports directory:", reportsDir);
//         }

//         // Generate PDF Report
//         const doc = new PDFDocument();
//         const stream = fs.createWriteStream(filePath);
//         doc.pipe(stream);

//         // 📝 Report Title
//         doc.fontSize(16).text(`Transaction Report for ${year}`, { align: "center" });
//         doc.moveDown();
//         doc.fontSize(12).text(`User ID: ${userId}`);
//         doc.moveDown();

//         // 🏦 **List All Transactions**
//         transactions.forEach((tx, index) => {
//             doc.text(`Transaction #${index + 1}`, { underline: true });
//             doc.text(`- Type: ${tx.type}`);
//             doc.text(`- Category: ${tx.category}`);
//             doc.text(`- Amount: $${tx.amount}`);
//             doc.text(`- Description: ${tx.description}`);
//             doc.text(`- Date: ${new Date(tx.date).toLocaleDateString()}`);
//             doc.moveDown();
//         });

//         doc.end();

//         stream.on("finish", () => {
//             console.log("PDF generation completed successfully!");
//             res.setHeader("Content-Type", "application/pdf");
//             res.setHeader("Content-Disposition", `inline; filename="tax-report-${userId}-${year}.pdf"`);
//             res.sendFile(filePath);
//         });

//         stream.on("error", (error) => {
//             console.error("Error writing PDF:", error);
//             res.status(500).json({ message: "Error generating report", error });
//         });

//     } catch (error) {
//         console.error("Error generating tax report:", error);
//         res.status(500).json({ message: "Error generating report", error });
//     }
// };

const fs = require("fs");
const path = require("path");
const PDFDocument = require("pdfkit");
const Transaction = require("../model/Transaction");

exports.generateTaxReport = async (req, res) => {
    try {
        const { userId, year } = req.params;
        console.log("Generating report for:", userId, "Year:", year);

        // Fetch transactions for the given user and year
        const transactions = await Transaction.find({
            user: userId,
            date: { 
                $gte: new Date(`${year}-01-01T00:00:00.000Z`), 
                $lte: new Date(`${year}-12-31T23:59:59.999Z`)
            }
        });

        if (transactions.length === 0) {
            return res.status(404).json({ message: "No transactions found for this user and year." });
        }

        // Define the directory and file path
        const reportsDir = path.join(__dirname, "../reports");
        const filePath = path.join(reportsDir, `tax-report-${userId}-${year}.pdf`);

        // Ensure the `reports` directory exists
        if (!fs.existsSync(reportsDir)) {
            fs.mkdirSync(reportsDir, { recursive: true });
            console.log("Created reports directory:", reportsDir);
        }

        // Generate PDF Report
        const doc = new PDFDocument({ margin: 40 });
        const stream = fs.createWriteStream(filePath);
        doc.pipe(stream);

        // **Header**
        doc
            .fontSize(18)
            .font("Helvetica-Bold")
            .fillColor("#0066cc")
            .text("Yearly Tax Report", { align: "center" })
            .moveDown(0.5);

        doc
            .fontSize(12)
            .fillColor("black")
            .text(`User ID: ${userId}`)
            .moveDown(0.5);

        doc
            .fontSize(12)
            .fillColor("black")
            .text(`Year: ${year}`)
            .moveDown(1);

        // **Table Header**
        doc
            .fontSize(12)
            .font("Helvetica-Bold")
            .fillColor("#333")
            .text("No.", 40, doc.y, { width: 50, align: "left" })
            .text("Category", 100, doc.y, { width: 150, align: "left" })
            .text("Amount ($)", 250, doc.y, { width: 100, align: "right" })
            .text("Date", 370, doc.y, { width: 150, align: "right" })
            .moveDown(0.5);

        doc.strokeColor("#cccccc").lineWidth(1).moveTo(40, doc.y).lineTo(550, doc.y).stroke().moveDown(0.5);

        // **Table Content**
        doc.font("Helvetica").fontSize(10);
        transactions.forEach((exp, index) => {
            doc
                .text(`${index + 1}`, 40, doc.y, { width: 50, align: "left" })
                .text(exp.category, 100, doc.y, { width: 150, align: "left" })
                .text(`$${exp.amount}`, 250, doc.y, { width: 100, align: "right" })
                .text(new Date(exp.date).toDateString(), 370, doc.y, { width: 150, align: "right" })
                .moveDown(0.5);

            doc.strokeColor("#dddddd").lineWidth(0.5).moveTo(40, doc.y).lineTo(550, doc.y).stroke().moveDown(0.2);
        });

        // **Footer**
        doc.moveDown(2);
        doc.fontSize(10).fillColor("#777").text("Generated by Expense Tracker", { align: "center" });

        doc.end();

        stream.on("finish", () => {
            res.setHeader("Content-Type", "application/pdf");
            res.setHeader("Content-Disposition", `inline; filename="tax-report-${userId}-${year}.pdf"`);
            res.sendFile(filePath);
        });

        stream.on("error", (error) => {
            console.error("Error writing PDF:", error);
            res.status(500).json({ message: "Error generating report", error });
        });

    } catch (error) {
        console.error("Error generating tax report:", error);
        res.status(500).json({ message: "Error generating report", error });
    }
};

