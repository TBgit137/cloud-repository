package com.xiaoyu.jdbc;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;

public class JDBCprac {
    public static void main(String[] args) throws Exception {
        // 1. Register the driver
        Class.forName("com.mysql.cj.jdbc.Driver");

        // 2. Establish the connection
        String url = "jdbc:mysql://localhost:3306/prac";
        String user = "root";
        String password = "MYSQLc-137040808";
        Connection conn = DriverManager.getConnection(url, user, password);

        // 3. Define the SQL statement
        String sql = "UPDATE `student gpa` SET gpa = 5.5 WHERE id = 1";

        // 4. Create a statement
        Statement stmt = conn.createStatement();

        try {
            // 5. Start the transaction
            conn.setAutoCommit(false);

            // 6. Execute the SQL statement
            int rowsAffected = stmt.executeUpdate(sql); // Rows affected

            // 7. Print the result
            System.out.println(rowsAffected);

            // 8. Commit the transaction
            conn.commit();
        } catch (Exception e) {
            // 9. Rollback the transaction
            conn.rollback();

            throw new RuntimeException(e);
        }

        // 10. Close the resources
        stmt.close();
        conn.close();
    }
}
